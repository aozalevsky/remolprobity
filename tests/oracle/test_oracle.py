"""
Oracle 1: stock MolProbity against remolprobity as one block (values
within 1e-6), stock's own clashscore included. Oracle 2: one block against
blocks of 5k atoms at margins 15, 20 and 25 Å and of 20k atoms at 20 Å
(values within 1e-9). An entry that a set's size would not split gets
blocks of a part of it instead (a quarter, an eighth or a half, at least
MIN_BLOCK_ATOMS), so every entry is compared across layouts: each run of
blocks must have more than one. A block whose margin is too small runs
again with the margin it asks for (the result's provenance lists them), so
every set must pass; some blocks may run again, never all.
An entry marked refused_by_both is a model stock MolProbity cannot take:
both oracles expect remolprobity to refuse it too, cleanly, and oracle 1
checks that stock does.
References are cached in the corpus directory, keyed by the model's SHA-256
and by what stock's result depends on in the image: the reduce, probe and
het dictionary hashes and the cctbx commit. Results are keyed by those, by
the SHA-256 of remolprobity's sources (remolprobity/**/*.py), and by the
parameters; a change to any of them runs them again.

REMOLPROBITY_ORACLE_NPROC: the workers of a run of blocks (default 4).
REMOLPROBITY_ORACLE_MAX_GB: the most memory a run may be expected to need,
and the scope it runs in (default 4: the workstation is shared). A run of
remolprobity whose plan expects more (run.expected_gb), or a stock
reference to make that expects more (20 KB per atom, plus 1 GB), is
skipped, saying what it would need.
"""
import functools
import glob
import hashlib
import json
import math
import os

import pytest

from helpers import REPO, in_image
from remolprobity import blocks, compare, io, result, run

CORPUS_DIR = os.environ.get('REMOLPROBITY_CORPUS', os.path.expanduser('~/.cache/remolprobity/corpus'))
ENTRIES = [e for e in json.load(open(os.path.join(REPO, 'tests', 'corpus', 'corpus.json')))
           if e.get('oracle', True) and e.get('tier', 'oracle') == 'oracle']
MAX_STOCK_ATOMS = 450000  # stock MolProbity needs about 20 KB per atom; 9A1N whole would take the workstation down
STOCK_BASE_GB, STOCK_GB_PER_ATOM = 1.0, 20e-6
NPROC = int(os.environ.get('REMOLPROBITY_ORACLE_NPROC', '4'))
MAX_GB = float(os.environ.get('REMOLPROBITY_ORACLE_MAX_GB') or 4)
SCOPE_GB = int(math.ceil(MAX_GB))
# oracle 2's sets: block atoms and margin, and the part of a model too small for that size to split
SETS = [(5000, 15, 4), (5000, 20, 4), (5000, 25, 8), (20000, 20, 2)]
MIN_BLOCK_ATOMS = 300
# sets in which no block of any entry ran again, at their own block size, unless the entry lists the set in
# retries_at; smaller blocks may need larger margins
NO_RETRIES = {(5000, 25), (20000, 20)}


class RunError(Exception):
    """A run in the image that failed; its stderr in full"""

    def __init__(self, what, returncode, stderr):
        super().__init__('%s failed (%d):\n%s' % (what, returncode, stderr[-4000:]))
        self.stderr = stderr


def path(entry):
    for ext in ('.cif', '.pdb'):
        p = os.path.join(CORPUS_DIR, entry['name'] + ext)
        if os.path.exists(p):
            return p
    pytest.skip('%s is not fetched (scripts/fetch_corpus.py)' % entry['name'])


def block_atoms(n_atoms, nominal, parts):
    """A set's block size for a model of n_atoms: its own where that splits the model, else a part of the model"""
    return nominal if n_atoms > nominal else max(MIN_BLOCK_ATOMS, n_atoms // parts)


@functools.lru_cache(maxsize=None)
def read(model):
    return io.read_model(model)


def _sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def image_state():
    """What results depend on in the image: reduce's, probe's and the het dictionary's hashes; the cctbx commit"""
    r = in_image('-c', 'import json\nfrom remolprobity import run, tools\n'
                       'print(json.dumps({"tools": run.tool_hashes(tools.find_tools()), "cctbx": run.cctbx_commit()}))')
    return json.loads(r.stdout.strip().splitlines()[-1])


def source_sha256():
    """SHA-256 over remolprobity's Python sources, with their paths"""
    package = os.path.join(REPO, 'remolprobity')
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(package, '**', '*.py'), recursive=True)):
        h.update(os.path.relpath(p, package).encode() + b'\0')
        with open(p, 'rb') as f:
            h.update(f.read() + b'\0')
    return h.hexdigest()


def reference_key():
    return _sha256_json(image_state())[:16]


def results_key():
    return _sha256_json(dict(image_state(), sources=source_sha256()))[:16]


def results_path(entry, label):
    return os.path.join(CORPUS_DIR, 'results', results_key(), '%s.%s.%s.pkl' % (entry['name'], entry['sha256'][:12],
                                                                             label))


def within_budget(what, gb):
    if gb > MAX_GB:
        pytest.skip('%s: expected peak %.1f GB, above REMOLPROBITY_ORACLE_MAX_GB=%g' % (what, gb, MAX_GB))


def same_image(provenance, what):
    state = image_state()
    assert provenance.get('tools') == state['tools'] and provenance.get('cctbx') == state['cctbx'], \
        '%s was made with other tools or cctbx: %s' % (what, {k: provenance.get(k) for k in state})


def refused(entry, model, *args):
    """Check that remolprobity refuses the model as the entry says: an error, no traceback, no result"""
    out = results_path(entry, 'refused')
    r = in_image('-m', 'remolprobity', model, '-o', out, '--geometry', 'v34,v33', *args,
                 check=False, timeout=3600, memory_gb=SCOPE_GB)
    assert r.returncode == 1, r.stderr[-4000:]
    assert entry['refused_by_both']['remolprobity'] in r.stderr and 'error:' in r.stderr, r.stderr[-4000:]
    assert 'Traceback' not in r.stderr and not os.path.exists(out), r.stderr[-4000:]


def ours(entry, model, size, margin=25, nproc=1):
    """remolprobity's result in one block (size 0) or in blocks of size atoms, run unless cached"""
    label = 'one' if size == 0 else '%d.%d' % (size, margin)
    out = results_path(entry, label)
    if not os.path.exists(out):
        m = read(model)
        plan = blocks.plan_blocks(m, size, margin)
        within_budget('remolprobity %s %s with %d workers' % (entry['name'], label, nproc),
                      run.choose_workers(plan.sizes(m), m.n_atoms, len(m.residues), nproc=nproc)[1])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        args = ['--block-atoms', str(size)] + (['--margin', str(margin), '--nproc', str(nproc)] if size else [])
        r = in_image('-m', 'remolprobity', model, '-o', out, '--geometry', 'v34,v33', *args,
                     check=False, timeout=6 * 3600, memory_gb=SCOPE_GB)
        if r.returncode != 0:
            raise RunError('remolprobity %s %s' % (os.path.basename(model), ' '.join(args)), r.returncode, r.stderr)
    r = result.read(out)
    same_image(r['provenance'], out)
    return r


def reference(entry, model):
    """Stock's result on the model, as a remolprobity result, made unless cached"""
    out = os.path.join(CORPUS_DIR, 'reference', '%s.%s.%s.pkl' % (entry['name'], entry['sha256'][:12],
                                                                  reference_key()))
    if not os.path.exists(out):
        atoms = read(model).n_atoms
        if atoms > MAX_STOCK_ATOMS:
            pytest.skip('%s has %d atoms: too many for stock MolProbity here' % (entry['name'], atoms))
        within_budget('stock MolProbity on %s' % entry['name'], STOCK_BASE_GB + STOCK_GB_PER_ATOM * atoms)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        in_image('-m', 'remolprobity', 'stock', model, '-o', out, '--workdir', out + '.work',
                 timeout=12 * 3600, memory_gb=SCOPE_GB)
    r = result.read(out)
    same_image(r['provenance'], out)
    return r


@pytest.mark.oracle
@pytest.mark.parametrize('entry', ENTRIES, ids=[e['name'] for e in ENTRIES])
def test_oracle1_stock_against_one_block(entry, tmp_path):
    model = path(entry)
    if entry.get('refused_by_both'):
        # stock as IHMValidation 3.4 runs it (as remolprobity stock does), on the file as fetched, since
        # remolprobity's reading refuses it
        r = in_image('-m', 'mmtbx.command_line.molprobity', 'disable_uc_volume_vs_n_atoms_check=True',
                     'coot=False', 'probe=False', 'prefix=%s' % (tmp_path / 'stock'), 'pickle=True', model,
                     check=False, timeout=3600, memory_gb=SCOPE_GB)
        assert r.returncode != 0 and entry['refused_by_both']['stock'] in r.stdout + r.stderr, r.stderr[-4000:]
        refused(entry, model, '--block-atoms', '0')
        return
    ref = reference(entry, model)
    one = ours(entry, model, 0)
    assert compare.differences(result.comparable(ref), result.comparable(one), tol=1e-6) == []
    # stock's own clashscore, which it counts before dropping clashes with the same id_str (the
    # reference's records are after); kept in the reference's provenance
    assert abs(ref['provenance']['clashscore'] - one['clashes']['clashscore']) <= 1e-6, \
        (ref['provenance']['clashscore'], one['clashes']['clashscore'])


@pytest.mark.oracle
@pytest.mark.parametrize('entry', ENTRIES, ids=[e['name'] for e in ENTRIES])
@pytest.mark.parametrize('nominal,margin,parts', SETS, ids=['%d-%d' % s[:2] for s in SETS])
def test_oracle2_blocks_against_one_block(entry, nominal, margin, parts):
    model = path(entry)
    if entry.get('refused_by_both'):
        refused(entry, model, '--block-atoms', str(nominal), '--margin', str(margin), '--nproc', str(NPROC))
        return
    size = block_atoms(read(model).n_atoms, nominal, parts)
    one = ours(entry, model, 0)
    many = ours(entry, model, size, margin, NPROC)
    plan = many['provenance']['blocks']
    assert plan['block_atoms'] == size and plan['margin'] == margin
    # a run of one block would compare one block with one block
    assert plan['count'] > 1, '%s in blocks of %d atoms is one block' % (entry['name'], size)
    retried = plan['retried']
    # some blocks, not all: blocks all running again would hide a broken margin check
    assert len(retried) < plan['count'], retried
    if size == nominal and (nominal, margin) in NO_RETRIES and \
            '%d/%d' % (nominal, margin) not in entry.get('retries_at', ()):
        assert retried == {}, retried
    assert compare.differences(result.comparable(one), result.comparable(many), tol=1e-9) == []
