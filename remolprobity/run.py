"""
remolprobity: MolProbity validation of a very large model in spatial blocks.

usage: mmtbx.python -m remolprobity MODEL -o RESULT [options]

RESULT: a name ending in .json, or '-' for stdout, gives JSON; any other
name a pickle.

Other commands:
  remolprobity stock MODEL -o RESULT   stock MolProbity on the whole model,
                                       as a result of the same form
  remolprobity compare A B             the differences between two results
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import __version__
from . import blocks, io, memory, result, schedule

DEFAULT_BLOCK_ATOMS = 130000
WORKER_BASE_GB, WORKER_GB_PER_ATOM = 0.3, 12e-6
# Every worker loads the compact model slice arrays (shared.pkl, ~20 B per residue)
WORKER_GB_PER_MODEL_RESIDUE = 0.0625e-6
MAIN_BASE_GB, MAIN_GB_PER_ATOM = 0.6, 6e-8
# When --gpu is enabled, a single persistent GPU server process holds the
# CUDA context, compiled XLA kernels and static tables, and the requests it
# runs at once: 1.7 GB RSS on 8WQL and 2.0 GB on 5Y6P, with 2 at once
GPU_SERVER_GB = 2.0
# a worker's problems with the data, reported by their message (the traceback kept in the workdir)
KNOWN = (io.InputError,)
PRELOAD = ('iotbx.pdb', 'mmtbx.model', 'mmtbx.validation.ramalyze', 'mmtbx.validation.rotalyze',
           'mmtbx.validation.clashscore', 'mmtbx.validation.restraints', 'remolprobity.worker')


def worker_gb(atoms, model_residues=0):
    """
    The expected peak of a worker on a block of this many atoms (core and
    margin), in a model of this many residues (GB)
    """
    return WORKER_BASE_GB + WORKER_GB_PER_ATOM * atoms + WORKER_GB_PER_MODEL_RESIDUE * model_residues


def main_gb(n_atoms, gpu=False):
    """The expected peak of the main process on a model of this many atoms (GB)"""
    return MAIN_BASE_GB + MAIN_GB_PER_ATOM * n_atoms + (GPU_SERVER_GB if gpu else 0.0)


def expected_gb(sizes, workers, n_atoms, n_residues, gpu=False):
    """
    The expected peak of a run (GB): its main process (and GPU server if enabled),
    and workers running the largest of these blocks (atoms each, core and margin) at once
    """
    return main_gb(n_atoms, gpu=gpu) + sum(worker_gb(s, n_residues) for s in sorted(sizes, reverse=True)[:workers])


def available_cpus():
    """CPUs this process may run on (its affinity: taskset, cpusets, SLURM), else the machine's"""
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def choose_workers(sizes, n_atoms, n_residues, memory_gb=None, nproc=None, cpus=1, gpu=False):
    """
    Workers for blocks of these sizes (atoms each, core and margin, as
    planned) and the expected peak (GB). nproc if given; else, with
    memory_gb, the most workers whose largest blocks fit it (at least one);
    else one per CPU. Never more than CPUs (unless given) or blocks.
    """
    if nproc is None:
        nproc = cpus
        if memory_gb is not None:
            nproc = min(nproc, len(sizes))
            while nproc > 1 and expected_gb(sizes, nproc, n_atoms, n_residues, gpu=gpu) > memory_gb:
                nproc -= 1
    nproc = max(1, min(nproc, len(sizes)))  # no idle workers
    return nproc, expected_gb(sizes, nproc, n_atoms, n_residues, gpu=gpu)


def file_sha256(path):
    """SHA-256 of a file, read in 1 MB chunks"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tool_hashes(t):
    """SHA-256 of the reduce and probe binaries and reduce's het dictionary"""
    het = t.reduce[t.reduce.index('-DB') + 1]
    return {'reduce_sha256': file_sha256(t.reduce[0]), 'probe_sha256': file_sha256(t.probe),
            'het_sha256': file_sha256(het)}


def cctbx_commit():
    try:
        import libtbx.load_env
        path = os.path.dirname(libtbx.env.dist_path('libtbx'))
        r = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'], capture_output=True, text=True)
        return r.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def parse_args(argv):
    p = argparse.ArgumentParser(prog='remolprobity', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model')
    p.add_argument('-o', '--output', required=True, help="the result: .json or '-' (stdout) for JSON, else a pickle")
    p.add_argument('--geometry', default='v34', help='comma-separated: v34, v33')
    p.add_argument('--memory', type=float,
                   help='GB for the whole run; chooses workers (default: 90%% of what the cgroup\'s memory limit '
                        'leaves or 80%% of what the machine has available, whichever is less)')
    p.add_argument('--block-atoms', type=int, help='core size; 0: the whole model in one block')
    p.add_argument('--margin', type=float, default=25.0,
                   help='Å around each core (default 25); a block that needs more is run again with what '
                        'it needs, up to twice this')
    p.add_argument('--nproc', type=int)
    p.add_argument('--reduce')
    p.add_argument('--probe')
    p.add_argument('--workdir')
    p.add_argument('--keep-workdir', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--block-timeout', type=float)
    p.add_argument('--zero-occupancy-as-one', action='store_true', help='an occupancy of 0, or none, as 1')
    p.add_argument('--missing-b-as-zero', action='store_true', help="a B of '.' or '?' (mmCIF) as 0")
    p.add_argument('--gpu', action='store_true', help='evaluate residue, geometry, and clash checks on GPU via JAX')
    return p.parse_args(argv)


def log(message):
    print('[remolprobity %s] %s' % (time.strftime('%H:%M:%S'), message), file=sys.stderr, flush=True)


def duration(seconds):
    """Seconds as the log gives them: '86 s', '3 min 05 s', '2 h 01 min'"""
    s = int(round(seconds))
    if s < 120:
        return '%d s' % s
    if s < 7200:
        return '%d min %02d s' % divmod(s, 60)
    return '%d h %02d min' % divmod(s // 60, 60)


def memory_summary(main_gb, blocks, at_once):
    """
    A run's peak memory, for the log. blocks: the peaks of each block's
    worker and of its reduce or probe (GB, the largest of them); at_once:
    workers. An estimate: the main process, and the blocks that need the
    most, as many as run at once, each with its reduce or probe, all at
    their peaks. Not a bound on what a memory-limited scope counts: that
    also has page cache (the image's files, the blocks) and apptainer.
    """
    at_once = min(at_once, len(blocks))
    total = main_gb + sum(sorted((w + c for w, c in blocks), reverse=True)[:at_once])
    return ('memory about %.2f GB without page cache: main process %.2f GB; %d block%s at a time, each up to %.2f GB, '
            'and up to %.2f GB more for reduce or probe'
            % (total, main_gb, at_once, '' if at_once == 1 else 's', max(w for w, _ in blocks),
               max(c for _, c in blocks)))


def describe(block_result):
    """A finished block for the log: atoms, seconds per stage (and what built its restraints), peak memory"""
    if 'margin_needed' in block_result:
        return ': needs a margin of %d Å (ran at %g)' % (block_result['margin_needed'], block_result['margin'])
    builders = block_result.get('geometry_builders', {})
    stages = []
    for stage, seconds in block_result['times'].items():
        builder = builders.get(stage[len('geometry_'):]) if stage.startswith('geometry_') else None
        stages.append('%s %.0fs%s' % (stage, seconds, ' (%s)' % builder if builder else ''))
    return ': %d atoms, %s, peak %.2f GB' % (block_result['atoms'], ', '.join(stages), block_result['peak_gb'])


def builder_counts(block_results):
    """{what built v34's restraints: blocks}; 'unknown' for a block resumed from a store that did not say"""
    counts = {'fast': 0, 'cctbx': 0}
    for br in block_results:
        builder = br.get('geometry_builders', {}).get('v34', 'unknown')
        counts[builder] = counts.get(builder, 0) + 1
    return counts


def retry_margins(block_results, margin_of, base):
    """
    {block: margin} for each block whose result asks for a larger margin (a
    MarginError in its worker): the margin it asks for. A block may ask for
    up to twice the run's margin (base), and only for more than it ran at
    (margin_of[block], else base); otherwise the run stops with the block's
    MarginError.
    """
    out = {}
    for br in block_results:
        if 'margin_needed' not in br:
            continue
        b, need, ran = br['block'], br['margin_needed'], margin_of.get(br['block'], base)
        if not ran < need <= 2 * base:
            raise schedule.BlockError(
                'block %d failed at a margin of %g Å:\nremolprobity.clashes.MarginError: %s\n(a block runs '
                'again with the margin it needs, up to twice --margin, %g Å)' % (b, ran, br['message'], 2 * base))
        out[b] = need
    return out


def retry_tasks(again, size, path, work, shared, members=None):
    """
    The tasks for the blocks run again ({block: margin}), one round for
    all of them, largest (size: atoms, core and margin) first. Each has its
    margin's block file (path[block]) and results store, results.m<margin>.
    """
    out = []
    for b in sorted(again, key=lambda b: (-size[b], b)):
        t = {'block': b, 'path': path[b], 'workdir': work, 'shared': shared, 'margin': again[b],
             'store': os.path.join(work, 'results.m%d' % again[b])}
        if members is not None:
            t['members'] = members[b]
        out.append(t)
    return out


def _plan_key(model, params):
    h = hashlib.sha256()
    h.update(model.sha256.encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()


def claim_stdout():
    """
    For a result written to stdout: the stream to write it to. Everything
    else that the run, its workers or its tools print to stdout goes to
    stderr from here on, so that stdout holds the result alone.
    """
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)
    return out


def inside(path, directory):
    """Whether path is directory or lies inside it, symbolic links and '..' resolved"""
    p, d = os.path.realpath(path), os.path.realpath(directory)
    return os.path.commonpath([p, d]) == d


def main(argv=None):
    a = parse_args(argv)
    if a.block_atoms is not None and a.block_atoms < 0:
        log('error: --block-atoms must be 0 (one block) or more, not %d' % a.block_atoms)
        return 1
    if a.memory is not None and not a.memory > 0:
        log('error: --memory must be more than 0 GB, not %g' % a.memory)
        return 1
    if a.workdir and os.path.exists(a.workdir) and not os.path.isdir(a.workdir):
        log('error: --workdir %s exists and is not a directory' % a.workdir)
        return 1
    if a.workdir and a.output != '-' and inside(a.output, a.workdir):
        # a run that succeeds removes the workdir, or what it made there
        log('error: -o %s is inside --workdir %s, which a successful run cleans up; write the result '
            'outside it' % (a.output, a.workdir))
        return 1
    a.stdout = claim_stdout() if a.output == '-' else None
    try:
        return _run(a)
    except (io.InputError, blocks.PlanError, schedule.BlockError, result.MergeError, RuntimeError,
            FileNotFoundError, ValueError) as e:
        log('error: %s' % e)
        return 1


def _run(a):
    from . import geometry, tools

    t0 = time.time()
    # before the model is read, so that what the cgroup has left does not count it
    budget = memory.budget(a.memory)
    definitions = [d.strip() for d in a.geometry.split(',') if d.strip()]
    for d in definitions:
        if d not in geometry.DEFINITIONS:
            raise ValueError('unknown geometry definition %r' % d)
    t = tools.find_tools(a.reduce, a.probe)
    log('remolprobity %s with %s' % (__version__, tools.versions(t)))
    log(budget.line)
    if budget.warning:
        log(budget.warning)
    log('reading %s' % a.model)
    model = io.read_model(a.model, zero_occupancy_as_one=a.zero_occupancy_as_one,
                          missing_b_as_zero=a.missing_b_as_zero)
    block_atoms = DEFAULT_BLOCK_ATOMS if a.block_atoms is None else a.block_atoms
    plan = blocks.plan_blocks(model, block_atoms, a.margin)
    sizes = plan.sizes(model)
    nproc, expected = choose_workers(sizes, model.n_atoms, len(model.residues), budget.gb, a.nproc,
                                     available_cpus(), gpu=a.gpu)
    log('%d atoms, %d residues: %d blocks (%.2fx the model, the largest %d atoms), %d workers, about %.1f GB'
        % (model.n_atoms, len(model.residues), plan.n_blocks, sum(sizes) / model.n_atoms, max(sizes), nproc,
           expected))
    if budget.gb is not None and expected > budget.gb:
        log('warning: expected peak %.1f GB exceeds the memory budget of %.1f GB; %s need less'
            % (expected, budget.gb, 'fewer workers (--nproc)' if nproc > 1 else 'smaller blocks (--block-atoms)'))

    created = a.workdir is None or not os.path.exists(a.workdir)  # ours to remove entirely on success
    if a.workdir and os.path.isdir(a.workdir) and os.listdir(a.workdir) \
            and not os.path.exists(os.path.join(a.workdir, 'plan.json')):
        raise RuntimeError('%s is not empty and holds no remolprobity run; use an empty or new directory'
                           % a.workdir)
    work = a.workdir or tempfile.mkdtemp(prefix='remolprobity.', dir=os.path.dirname(os.path.abspath(a.output)))
    os.makedirs(work, exist_ok=True)
    log('workdir %s (kept if the run fails once its blocks are written)' % work)
    found = set(os.listdir(work))
    try:
        return _run_in(a, t, model, plan, sizes, nproc, budget.gb, block_atoms, definitions, work, created, t0)
    except BaseException:
        if not os.path.exists(os.path.join(work, 'plan.json')):
            # stopped before its plan was saved: nothing to resume, so leave
            # the directory as it was found (or gone, if the run made it)
            _clean(work, created, found)
        raise


def _clean(work, created, found):
    """
    Remove the workdir if the run made it, else what the run added to it (all
    but found). It runs while an error is on its way out, so it never raises:
    what it cannot remove stays.
    """
    if created:
        shutil.rmtree(work, ignore_errors=True)
        return
    try:
        names = set(os.listdir(work)) - found
    except OSError:
        return
    for name in names:
        path = os.path.join(work, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def _run_in(a, t, model, plan, sizes, nproc, budget_gb, block_atoms, definitions, work, created, t0):
    """The run, once its workdir exists"""
    from . import naming, tools, worker

    gpu_srv = None
    if a.gpu:
        from . import jax_tables
        gpu_srv = jax_tables.GPUServer(work).start()
        log('GPU server started on %s (%s)' % (gpu_srv.info['backend'], gpu_srv.info['device']))
    try:
        # v34 restraints from the fast builder where it takes a block: always with --gpu, and in CPU
        # runs wherever JAX is installed (as in remolprobity's image), evaluated in the workers
        fast_v34 = 'v34' in definitions and (a.gpu or importlib.util.find_spec('jax') is not None)
        tables = None
        if gpu_srv is not None:
            tables = gpu_srv.info['tables']
        elif fast_v34:
            from . import jax_tables
            tables = jax_tables.table_hashes()
        hashes, commit = tool_hashes(t), cctbx_commit()
        # whatever can change a block's result: input, parameters, tools, code
        params = {'block_atoms': block_atoms, 'margin': a.margin, 'geometry': definitions,
                  'zero_occupancy_as_one': a.zero_occupancy_as_one, 'missing_b_as_zero': a.missing_b_as_zero,
                  'gpu': a.gpu, 'reduce': t.reduce, 'probe': t.probe,
                  'tools': hashes, 'cctbx': commit, 'remolprobity': __version__}
        if tables is not None:
            params['jax_tables'] = tables
        key = _plan_key(model, params)
        plan_file = os.path.join(work, 'plan.json')
        if os.path.exists(plan_file):
            if not a.resume:
                raise RuntimeError('%s holds a previous run; use --resume or another --workdir' % work)
            with open(plan_file) as f:
                previous = json.load(f)
            if previous['key'] != key:
                raise RuntimeError('the input, the parameters, the tools or the code changed since the run in %s'
                                   % work)
            created = previous.get('created', created)
        elif a.resume:
            raise RuntimeError('nothing to resume in %s' % work)

        check = tools.self_check(t, os.path.join(work, 'scratch'))
        # only a run asked for as one block may go on: a default run that
        # happens to fit one block would change behaviour with the model's size
        if not check['passed'] and block_atoms != 0:
            raise RuntimeError('reduce failed the self-check (orientation %s vs %s): its results depend on '
                               'unrelated parts of the model, so blocks would not match a whole-model run; '
                               'use a reduce with the clique-order patch, or --block-atoms 0'
                               % (check['near'], check['far']))

        block_dir = os.path.join(work, 'blocks')
        os.makedirs(block_dir, exist_ok=True)
        ext = '.cif' if model.format == 'cif' else '.pdb'
        paths = [os.path.join(block_dir, 'block%05d%s' % (b, ext)) for b in range(plan.n_blocks)]
        if not os.path.exists(plan_file):
            if model.spans is not None:
                io.check_segments(model, plan.members, paths)
            else:
                paths = io.write_blocks(model, plan.members, block_dir, a.zero_occupancy_as_one,
                                        missing_b_as_zero=a.missing_b_as_zero)
            with open(plan_file, 'w') as f:
                json.dump({'key': key, 'blocks': plan.n_blocks, 'created': created}, f)
        shared = os.path.join(work, 'shared.pkl')
        mslice = io.model_slice(model, a.zero_occupancy_as_one, a.missing_b_as_zero)
        with open(shared, 'wb') as f:
            pickle.dump({'chain_ids': mslice['auth_table'],
                         'res_chain_idx': mslice['res_auth_idx'],
                         'res_ids': np.array([rid for _, rid in model.residues], dtype='S'),
                         'owner_arr': np.asarray(plan.owner, dtype=np.int32),
                         'stretch_arr': np.asarray(io.stretches(model), dtype=np.int32),
                         'model_slice': mslice,
                         'conversion': naming.conversion(model.chain_ids, model.residue_names),
                         'occupancy_max': 1.0 if a.zero_occupancy_as_one and not model.occupancy_max
                         else model.occupancy_max,
                         'tools': t, 'definitions': definitions, 'margin': a.margin, 'gpu': a.gpu,
                         'fast_v34': fast_v34,
                         'gpu_sock': gpu_srv.sock_path if gpu_srv is not None else None}, f, protocol=4)
        order = sorted(range(plan.n_blocks), key=lambda b: -sizes[b])
        tasks = [{'block': b, 'path': paths[b], 'workdir': work, 'shared': shared,
                  'members': plan.members[b]} for b in order]
        block_results = schedule.run_tasks(worker.run_block, tasks, nproc, timeout=a.block_timeout,
                                           store=os.path.join(work, 'results'), log=log, preload=PRELOAD,
                                           describe=describe, known=KNOWN)
        # A block whose core needs more margin than it has (a reduce clique or a
        # clash reaching too near its edge) runs again with what it asks for:
        # same core, so the same residues owned, and its own checks certify it.
        # Each margin has its own block files and results store.
        retried, members = {}, list(plan.members)
        atom_owner = plan.owner[model.res_of]
        residue_atoms = model.atoms_per_residue()
        at = {br['block']: i for i, br in enumerate(block_results)}
        while True:
            again = retry_margins(block_results, retried, a.margin)
            if not again:
                break
            todo = sorted(again)
            for b in todo:
                members[b] = blocks.block_members(model, plan.owner, b, again[b], atom_owner)
            blocks.check_neighbours(blocks.Plan(plan.owner, members, a.margin, block_atoms), model)
            # larger than the first round planned for: say how large, and warn against the budget
            size = {b: int(residue_atoms[members[b]].sum()) for b in todo}
            for b in todo:
                log('block %d needs a margin of %d Å: running it again, %d atoms, about %.1f GB'
                    % (b, again[b], size[b], worker_gb(size[b], len(model.residues))))
            # all of them in one round, with up to nproc workers, fewer if their larger blocks need it
            workers, peak = choose_workers(list(size.values()), model.n_atoms, len(model.residues), budget_gb,
                                           cpus=nproc, gpu=a.gpu)
            if budget_gb is not None and peak > budget_gb:
                log('warning: blocks run again with larger margins may need %.1f GB, more than the memory budget of '
                    '%.1f GB' % (peak, budget_gb))
            if model.spans is not None:
                path = {b: os.path.join(block_dir, 'block%05d.m%d%s' % (b, again[b], ext)) for b in todo}
                io.check_segments(model, [members[b] for b in todo], [path[b] for b in todo])
            else:
                path = dict(zip(todo, io.write_blocks(model, [members[b] for b in todo], block_dir,
                                                      a.zero_occupancy_as_one,
                                                      names=['block%05d.m%d' % (b, again[b]) for b in todo],
                                                      missing_b_as_zero=a.missing_b_as_zero)))
            for br in schedule.run_tasks(worker.run_block, retry_tasks(again, size, path, work, shared, members),
                                         workers, timeout=a.block_timeout, log=log, preload=PRELOAD,
                                         describe=describe, known=KNOWN):
                block_results[at[br['block']]] = br
            retried.update(again)
        gpu_peak_gb = gpu_srv.stop() if gpu_srv is not None else 0.0
    finally:
        if gpu_srv is not None:
            gpu_srv.stop()
    peak_gb = {'main': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
               'worker': max(br['peak_gb'] for br in block_results)}
    if gpu_srv is not None:
        peak_gb['gpu_server'] = gpu_peak_gb
    provenance = {
        'remolprobity': __version__, 'cctbx': commit, 'tools': hashes,
        'reduce': t.reduce_version, 'probe': t.probe_version, 'reduce_self_check': check,
        'input': {'path': os.path.abspath(a.model), 'sha256': model.sha256, 'atoms': model.n_atoms,
                  'residues': len(model.residues), 'chains': len(model.chain_ids),
                  'hydrogens': model.hydrogens, 'zero_occupancy': model.zero_occupancy},
        'blocks': {'count': plan.n_blocks, 'block_atoms': block_atoms, 'margin': a.margin,
                   'overhead': plan.overhead(model), 'workers': nproc,
                   'retried': {int(b): int(m) for b, m in sorted(retried.items())}},
        'geometry': definitions, 'deviations': list(result.DEVIATIONS if check['passed'] else result.DEVIATIONS[1:]),
        'options': {'zero_occupancy_as_one': a.zero_occupancy_as_one, 'missing_b_as_zero': a.missing_b_as_zero,
                    'gpu': a.gpu},
        'wall_seconds': time.time() - t0,
        'peak_gb': peak_gb,
    }
    if gpu_srv is not None:
        provenance['gpu'] = gpu_srv.info
    if 'v34' in definitions:
        provenance['v34_builder'] = {'blocks': builder_counts(block_results), 'tables': tables}
    r = result.merge(model, block_results, definitions, provenance)
    result.write(r, a.output, out=a.stdout)
    log('done in %s: clashscore %.2f, %d Ramachandran outliers, %d rotamer outliers'
        % (duration(time.time() - t0), r['clashes']['clashscore'], r['ramachandran']['outliers'],
           r['rotamers']['outliers']))
    # a block resumed from the store of an earlier remolprobity may lack its reduce's and probe's peak
    log(memory_summary(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6 + gpu_peak_gb,
                       [(br['peak_gb'], br.get('children_peak_gb', 0.0)) for br in block_results], nproc))
    if not a.keep_workdir:
        if created:
            shutil.rmtree(work)
        else:  # a directory of the user's: only what this run made
            stores = sorted(n for n in os.listdir(work) if re.fullmatch(r'results\.m\d+', n))
            for name in ['plan.json', 'shared.pkl', 'blocks', 'results', 'scratch'] + stores:
                path = os.path.join(work, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.exists(path):
                    os.remove(path)
    return 0
