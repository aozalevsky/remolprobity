"""Clashes in a block: reduce, then probe, as MolProbity's clashscore runs them"""
import math
import os
import re
import shlex
import string
import subprocess

import numpy as np

from .ids import residue_id
from .io import InputError
from .naming import pdb_copy
from .tools import add_hydrogens

REACH = 5.0  # Å: a contact's reach, for checking the margin holds a block's partners
# A group in reduce's USER  MOD records (its descriptor, between the first and
# second ':'): chain id (2 characters), residue number, insertion code, then
# residue name (3), atom name (4) and altloc (1). reduce writes the number in
# hybrid-36 (4 characters, A000 for 10000) for groups it rotates, but in
# decimal of any width for groups it flips (amide, His): 10742, not A0LQ.
_GROUP = re.compile(r'(.{2})(\s*-?\d+|[A-Za-z][0-9A-Za-z]{3})(.).{8}')


class MarginError(Exception):
    """
    A block whose margin doesn't hold what its core needs. needed: a
    margin (Å) that would hold it, or None when no margin would
    """

    def __init__(self, message, needed=None):
        super().__init__(message)
        self.needed = needed


def probe_lines(command, pdb_file):
    """probe's output, one line at a time, without the newline (command ends with ' -')"""
    command = command.rstrip()
    assert command.endswith(' -')
    command = command[:-1] + shlex.quote(pdb_file)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1 << 20)
    for line in p.stdout:
        yield line.rstrip('\n')
    err = p.stderr.read()
    if p.wait() != 0:
        raise RuntimeError('probe failed:\n%s' % err)


def _line_key(line):
    return (line[20:22].strip(), line[22:26].strip(), line[26:27].strip(),
            line[17:20].strip(), line[12:16].strip(), line[16:17].strip())


def _atom_key(a):
    return (a.chain_id.strip(), a.resseq.strip(), a.icode.strip(),
            a.resname.strip(), a.name.strip(), a.altloc.strip())


def _coordinates(withh):
    """Coordinates of the protonated block's atoms by PDB key, the first if repeated"""
    found = {}
    with open(withh) as f:
        for line in f:
            if line.startswith(('ATOM  ', 'HETATM')):
                found.setdefault(_line_key(line), (float(line[30:38]), float(line[38:46]),
                                                   float(line[46:54])))
    return found


def _by_residue(xyz):
    """The points of _coordinates by residue: {(chain, resid): [xyz, ...]}, in PDB chain ids"""
    out = {}
    for k, p in xyz.items():
        out.setdefault((k[0], residue_id(k[1], k[2])), []).append(p)
    return out


def _cliques(withh):
    """
    reduce's cliques, from its USER MOD Set records: lists of members
    (chain, resid), in PDB chain ids. A record whose group has no residue
    number where one belongs is an error, not a member skipped.
    """
    sets = {}
    with open(withh) as f:
        for line in f:
            if line.startswith('USER  MOD Set'):
                head, rest = line[len('USER  MOD Set'):].split(':', 1)
                group = _GROUP.fullmatch(rest.split(':', 1)[0])
                if group is None:
                    raise RuntimeError("reduce's record %r: no residue number where reduce writes one"
                                       % line.rstrip())
                sets.setdefault(int(head.split('.')[0]), []).append(
                    (group.group(1).strip(), residue_id(group.group(2), group.group(3))))
    return list(sets.values())


def check_cliques(cliques, points_of, core_tree, margin, label):
    """
    Raise MarginError for a clique with a member within REACH of the core
    (an owned member, or one an owned atom can touch) that has any point
    beyond margin - REACH of it: reduce could orient it differently with
    its surroundings cut off. points_of(member) lists the member's xyz;
    label(member) names it for messages. A member without points is an
    error: its clique could not be checked.
    """
    if core_tree is None:
        return
    limit = margin - REACH
    for clique in cliques:
        points = []
        for m in clique:
            found = points_of(m)
            if not len(found):
                raise RuntimeError("reduce's clique member %s has no atoms in the block it was found in"
                                   % label(m))
            points.extend(found)
        d, _ = core_tree.query(np.asarray(points))
        if float(np.min(d)) > REACH:
            continue
        far = float(np.max(d))
        if far > limit:
            needed = math.ceil(far + REACH)
            raise MarginError('a reduce clique near the core (%s) reaches within %.0f Å of the margin; '
                              'use --margin %d or more'
                              % (', '.join(label(m) for m in clique), REACH, needed), needed)


SPARE_IDS = [a + b for a in ' ' + string.ascii_letters + string.digits for b in string.ascii_letters + string.digits]


def spare_chain_ids(hierarchy, stretch_of):
    """
    Chain ids for reduce's input such that two neighbouring residues of the
    block with one chain id but from different stretches of the model do not
    look like one chain to reduce (it joins neighbours with one id). Returns
    (renamed, restore): renamed[(chain, resseq, icode)] is the new id of each
    residue (PDB names, stripped) that gets one, the rest of its stretch in
    the block sharing it; restore[new id] is the old id. A stretch may lie in
    part of a chain object, or a chain object may hold several.
    stretch_of(chain, resseq, icode) gives a residue's stretch.
    """
    used = {c.id.strip() for c in hierarchy.models()[0].chains()}
    spare = (i.strip() for i in SPARE_IDS if i.strip() not in used)
    renamed, restore = {}, {}
    prev = prev_out = prev_stretch = None
    for c in hierarchy.models()[0].chains():
        orig = c.id.strip()
        for rg in c.residue_groups():
            key = (orig, rg.resseq.strip(), rg.icode.strip())
            s = stretch_of(*key)
            out = orig
            if prev == orig and s == prev_stretch:
                out = prev_out
            elif prev == orig:
                try:
                    out = next(spare)
                except StopIteration:
                    raise InputError('no spare chain id for reduce: the block has too many chain ids')
            if out != orig:
                renamed[key] = out
                restore[out] = orig
            prev, prev_out, prev_stretch = orig, out, s
    return renamed, restore


def rename_chain_ids(pdb_text, renamed):
    """The PDB text with the chain id of each residue in renamed ({(chain, resseq, icode): new id}) replaced"""
    out = []
    for line in pdb_text.splitlines(keepends=True):
        if line.startswith(('ATOM  ', 'HETATM', 'ANISOU')):
            new = renamed.get((line[20:22].strip(), line[22:26].strip(), line[26:27].strip()))
            if new is not None:
                line = line[:20] + '%2s' % new + line[22:]
        out.append(line)
    return ''.join(out)


def restore_chain_ids(path, restore):
    """Rewrite reduce's output at path with the original chain ids: atom lines and USER  MOD Set/Single records"""
    if not restore:
        return
    tmp = path + '.restored'
    with open(path) as fi, open(tmp, 'w') as fo:
        for line in fi:
            if line.startswith(('ATOM  ', 'HETATM', 'ANISOU')):
                old = restore.get(line[20:22].strip())
                if old is not None:
                    line = line[:20] + '%2s' % old + line[22:]
            elif line.startswith(('USER  MOD Set', 'USER  MOD Single')):
                i = line.find(':', 14)  # after the set number, 'Set 2.1:' or 'Single :'
                old = restore.get(line[i + 1:i + 3].strip())
                if old is not None:
                    line = line[:i + 1] + '%2s' % old + line[i + 3:]
            fo.write(line)
    os.replace(tmp, path)


def block_clashes(noh, conv, owned, core_tree, margin, tools, workdir, occupancy_max, stretch=None):
    """
    The clashes this block owns, as (atom1, atom2, overlap, distance) in the
    model's names, and the atoms probe counts in its core. noh: the block
    without hydrogens, symmetric side chains flipped; owned(chain, resid);
    core_tree: a cKDTree of the core's atoms. A clash is owned by the block
    owning its first atom in probe's order. Raises MarginError when a reduce
    clique with an owned group, or an owned clash, comes within REACH of
    the margin's outer edge. stretch: {(chain, resid): k} of the whole model's
    stretches (io.stretches); reduce sees the block with chain ids made
    distinct where blocks joined stretches that the model keeps apart, and its
    output is restored to the global names before anything else reads it.
    """
    from mmtbx.validation import utils
    from mmtbx.validation.clashscore import probe_clashscore_manager

    h, names = pdb_copy(noh, conv)

    def model_key(chain, resseq, icode):
        c, r, _ = names[(chain.strip(), resseq.strip(), icode.strip())]
        return c, r

    # a residue's altlocs may be different residue types (microheterogeneity):
    # names holds the first group's name, so map each atom's own name back
    resnames = {}
    for ag_model, ag_pdb in zip(noh.atom_groups(), h.atom_groups()):
        resnames[ag_pdb.resname.strip()] = ag_model.resname.strip()

    def named(a):
        c, r, _ = names[(a.chain_id.strip(), a.resseq.strip(), a.icode.strip())]
        n = a.resname.strip()
        return (c, r, resnames.get(n, n), a.name.strip(), a.altloc.strip())

    noh_pdb = os.path.join(workdir, 'noh.pdb')
    restore = {}
    text = h.as_pdb_string()
    if stretch is not None:
        renamed, restore = spare_chain_ids(h, lambda c, r, i: stretch[names[(c, r, i)][:2]])
        text = rename_chain_ids(text, renamed)
    with open(noh_pdb, 'w') as f:
        f.write(text)
    use_segids = utils.use_segids_in_place_of_chainids(hierarchy=h)
    model_id = h.models()[0].id
    del h
    withh = add_hydrogens(tools, noh_pdb, workdir)
    os.remove(noh_pdb)
    restore_chain_ids(withh, restore)
    xyz = _coordinates(withh)
    by_residue = _by_residue(xyz)
    limit = margin - REACH

    def farthest(points):
        """The largest distance from the core to the points, None without a core"""
        if core_tree is None:
            return None
        d, _ = core_tree.query(np.asarray(points))
        return float(np.max(d))

    def too_far(points):
        """(True, advice, margin that would hold them or None) when the points reach past the margin's inner limit"""
        far = farthest(points)
        if far is None:
            return True, 'the block owns nothing near it', None
        if far > limit:
            needed = math.ceil(far + REACH)
            return True, 'use --margin %d or more' % needed, needed
        return False, None, None

    # members and atoms both as (chain, resid): reduce's members can be
    # decimal where its atoms are hybrid-36
    model_residue = {(c, residue_id(r, i)): v[:2] for (c, r, i), v in names.items()}
    check_cliques(_cliques(withh), lambda m: by_residue.get(m, []), core_tree, margin,
                  lambda m: '%s %s' % model_residue.get(m, m))

    pcm = probe_clashscore_manager(h_pdb_string=None, fast=False, condensed_probe=True, nuclear=False,
                                   largest_occupancy=occupancy_max, use_segids=use_segids,
                                   model_id=model_id)
    probe = shlex.quote(tools.probe)
    found = pcm.process_raw_probe_output(probe_lines(pcm.probe_txt.replace(pcm.probe_command, probe), withh))
    seen, records = set(), []
    for c in sorted(found):
        a1, a2 = c.atoms_info
        key = c.id_str()
        if key in seen or not owned(*model_key(a1.chain_id, a1.resseq, a1.icode)):
            continue
        seen.add(key)
        p1, p2 = xyz[_atom_key(a1)], xyz[_atom_key(a2)]
        bad, advice, needed = too_far([p1, p2])
        if bad:
            raise MarginError('the clash %s - %s reaches within %.0f Å of the margin; %s'
                              % (named(a1), named(a2), REACH, advice), needed)
        records.append((named(a1), named(a2), abs(c.overlap), math.dist(p1, p2)))

    core_pdb = os.path.join(workdir, 'core.pdb')
    with open(withh) as fi, open(core_pdb, 'w') as fo:
        for line in fi:
            if line.startswith(('ATOM  ', 'HETATM')) and owned(*model_key(line[20:22], line[22:26], line[26:27])):
                fo.write(line)
    n = 0
    for line in probe_lines(pcm.probe_atom_txt.replace(pcm.probe_command, probe), core_pdb):
        try:
            n = int(line.split(':')[1])
        except (IndexError, ValueError):
            pass
    os.remove(withh)
    os.remove(core_pdb)
    return {'records': records, 'probe_atoms': n}
