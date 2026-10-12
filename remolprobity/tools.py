"""
reduce and probe: where they are, running reduce, and the reduce self-check;
and the versions of what computes the numbers
"""
import os
import re
import subprocess
from dataclasses import dataclass

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
# A serine's hydroxyl in a 12 Å sphere of 9A1N, and in the same sphere with
# 25 residues 500 Å away. Stock reduce orients it differently in the two:
# its clique members come out in an order that depends on the other
# cliques. With the clique-order patch they don't.
SELF_CHECK = ('selfcheck_near.pdb', 'selfcheck_far.pdb', ('24', '276', 'SER', 'OG'))


@dataclass
class Tools:
    reduce: list
    probe: str
    reduce_version: str
    probe_version: str


def _version(cmd, what):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except OSError as e:
        raise RuntimeError('cannot run %s (%s): %s' % (what, cmd[0], e.strerror or e))
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    return lines[0] if lines else 'unknown'


def find_tools(reduce=None, probe=None):
    """The MolProbity installation's reduce (with its het dictionary) and probe, unless given"""
    import libtbx.load_env

    het = libtbx.env.under_dist('reduce', 'reduce_wwPDB_het_dict.txt')
    reduce = reduce or libtbx.env.under_build('reduce/exe/reduce')
    probe = probe or libtbx.env.under_build('probe/exe/probe')
    for what, path in (('reduce', reduce), ('probe', probe)):
        if not os.path.isfile(path):
            raise RuntimeError('%s not found: %s (give it with --%s)' % (what, path, what))
        if not os.access(path, os.X_OK):
            raise RuntimeError('%s is not executable: %s' % (what, path))
    if not os.path.isfile(het):
        raise RuntimeError("reduce's het dictionary not found: %s" % het)
    return Tools([reduce, '-DB', het, '-ALLALT'], probe,
                 _version([reduce, '-version'], 'reduce'), _version([probe, '-version'], 'probe'))


def _git(path, *args):
    """git's answer for the checkout at path (its own, not one around it); None if there is none"""
    try:
        top = subprocess.run(['git', '-C', path, 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
        if top.returncode != 0 or os.path.realpath(top.stdout.strip()) != os.path.realpath(path):
            return None
        r = subprocess.run(['git', '-C', path] + list(args), capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def molprobity_version(root):
    """'MolProbity VERSION (COMMIT)' for MolProbity's checkout at root, as much as is known; None if none"""
    try:
        with open(os.path.join(root, 'lib', 'core.php')) as f:
            m = re.search(r'^\s*define\("MP_VERSION",\s*"([^"]+)"', f.read(), re.M)
    except OSError:
        return None
    commit = _git(root, 'rev-parse', '--short=12', 'HEAD')
    if m and commit:
        return 'MolProbity %s (%s)' % (m.group(1), commit)
    return 'MolProbity %s' % (m.group(1) if m else commit or 'unknown')


def tool_version(name, version):
    """reduce's or probe's -version, read plainly: 'reduce.4.16.250520' as 'reduce 4.16.250520'"""
    v = version.strip()
    if v.lower().startswith(name):
        v = v[len(name):].lstrip('.: ')
    return '%s %s' % (name, v or 'unknown')


def versions(t):
    """What computes the numbers, for the log: MolProbity, cctbx, reduce and probe"""
    import libtbx.load_env

    cctbx = os.path.dirname(libtbx.env.dist_path('libtbx'))
    try:
        from libtbx.version import get_version
        cctbx_version = get_version()
    except Exception:
        cctbx_version = _git(cctbx, 'rev-parse', '--short=12', 'HEAD') or 'unknown'
    # bootstrap.py --builder=molprobity puts MolProbity's own checkout beside modules/
    mp = molprobity_version(os.path.join(os.path.dirname(os.path.dirname(cctbx)), 'molprobity'))
    return ', '.join(([mp] if mp else []) + ['cctbx %s' % cctbx_version, tool_version('reduce', t.reduce_version),
                                             tool_version('probe', t.probe_version)])


def _last_line(data):
    """The last line of data (bytes) that is not blank"""
    lines = data.rstrip().rsplit(b'\n', 1)
    return lines[-1].strip()


def run_reduce(tools, args, fin, fout):
    """
    reduce on fin into fout. reduce's exit status is not 0 on success (255
    after -trim, 1 when it gives up on a clique), so success is: an exit,
    not a signal; output to its end; atoms written. reduce copies an END
    record through, last: the input gets one if it has none (nothing else
    in the output changes), and the output must end with it, since reduce
    stops writing on a full disk without an error.
    """
    with open(fin, 'rb') as i:
        data = i.read()
    if _last_line(data) != b'END':
        data = data + (b'' if data.endswith(b'\n') or not data else b'\n') + b'END\n'
    with open(fout, 'wb') as o:
        r = subprocess.run(tools.reduce + args, input=data, stdout=o, stderr=subprocess.PIPE)
    what = 'reduce %s on %s' % (' '.join(args), fin)
    err = r.stderr.decode(errors='replace')[-2000:]
    if r.returncode < 0:
        raise RuntimeError('%s was killed by signal %d:\n%s' % (what, -r.returncode, err))
    with open(fout, 'rb') as f:
        f.seek(max(0, os.path.getsize(fout) - 4096))
        complete = _last_line(f.read()) == b'END'
    with open(fout) as f:
        if not any(line.startswith(('ATOM', 'HETATM')) for line in f):
            raise RuntimeError('%s wrote no atoms:\n%s' % (what, err))
    if not complete:
        raise RuntimeError('%s wrote incomplete output (no END record at its end; is the disk full?):\n%s'
                           % (what, err))


def add_hydrogens(tools, noh_pdb, workdir, time_limit=120):
    """The model in noh_pdb with hydrogens as MolProbity's clashscore places them: trim, then build without flips"""
    trimmed = os.path.join(workdir, 'trimmed.pdb')
    withh = os.path.join(workdir, 'withh.pdb')
    run_reduce(tools, ['-quiet', '-trim', '-'], noh_pdb, trimmed)
    run_reduce(tools, ['-oh', '-his', '-flip', '-keep', '-allalt', '-limit%d' % time_limit,
                       '-pen9999', '-'], trimmed, withh)
    os.remove(trimmed)
    return withh


def _rotation(withh, target):
    tag = ':%2s%4s %s %-4s' % target
    with open(withh) as f:
        for line in f:
            if line.startswith('USER  MOD') and tag in line:
                return line.split('rot')[1].split(':')[0].strip()
    return None


def self_check(tools, workdir):
    """Whether reduce orients the self-check serine the same with and without far residues"""
    near, far, target = SELF_CHECK
    rotations = []
    for name in (near, far):
        d = os.path.join(workdir, 'selfcheck_' + name[:-4])
        os.makedirs(d, exist_ok=True)
        rotations.append(_rotation(add_hydrogens(tools, os.path.join(DATA, name), d), target))
    return {'passed': rotations[0] is not None and rotations[0] == rotations[1],
            'near': rotations[0], 'far': rotations[1]}
