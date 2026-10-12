import json
import os
import pickle
import subprocess

import pytest

from helpers import DATA, REPO, STOCK_REDUCE, in_image
from remolprobity import compare, result

L_L10 = os.path.join(DATA, '9a1n_L_L10.cif')


def remolprobity(*args, check=True):
    return in_image('-m', 'remolprobity', *args, check=check, timeout=3600)


@pytest.mark.container
def test_blocks_agree_with_one_block(tmp_path):
    one, many = str(tmp_path / 'one.pkl'), str(tmp_path / 'many.pkl')
    remolprobity(L_L10, '-o', one, '--block-atoms', '0', '--geometry', 'v34,v33')
    remolprobity(L_L10, '-o', many, '--block-atoms', '2500', '--margin', '20', '--nproc', '4',
                 '--geometry', 'v34,v33')
    a, b = result.read(one), result.read(many)
    assert b['provenance']['blocks']['count'] > 2
    assert compare.differences(result.comparable(a), result.comparable(b), tol=1e-9) == []
    assert a['clashes']['records'] and a['ramachandran']['total'] > 1000
    # no angle with the same atom twice (what cctbx's NCS shortcut used to make)
    for d in ('v34', 'v33'):
        assert all(len(set(o[:3])) == 3 for o in a['geometry'][d]['angles']['outliers'])


@pytest.mark.container
def test_split_chain_blocks_agree(tmp_path):
    # Review Focus 3
    p = os.path.join(DATA, '9a1n_split_chain.cif')
    one, many = str(tmp_path / 'one.pkl'), str(tmp_path / 'many.pkl')
    remolprobity(p, '-o', one, '--block-atoms', '0')
    remolprobity(p, '-o', many, '--block-atoms', '2500', '--nproc', '4')
    assert compare.differences(result.comparable(result.read(one)), result.comparable(result.read(many))) == []


@pytest.mark.container
def test_rna_only_model(tmp_path):
    # Review Focus 2: no protein, fewer residues than workers
    out = str(tmp_path / 'r.pkl')
    remolprobity(os.path.join(DATA, '1ehz_10nt.cif'), '-o', out, '--nproc', '8')
    r = result.read(out)
    assert r['ramachandran']['total'] == 0 and r['rotamers']['total'] == 0
    assert r['clashes']['atoms'] > 0 and r['clashes']['clashscore'] >= 0


@pytest.mark.container
def test_resume_refuses_changed_parameters(tmp_path):
    # Review Focus 4
    work, out = str(tmp_path / 'work'), str(tmp_path / 'r.pkl')
    remolprobity(L_L10, '-o', out, '--block-atoms', '2500', '--workdir', work, '--keep-workdir')
    r = remolprobity(L_L10, '-o', out, '--block-atoms', '2500', '--margin', '30', '--workdir', work,
                     '--resume', check=False)
    assert r.returncode == 1 and 'changed' in r.stderr


@pytest.mark.container
def test_resume_reuses_stored_blocks(tmp_path):
    work, out = str(tmp_path / 'work'), str(tmp_path / 'r.pkl')
    remolprobity(L_L10, '-o', out, '--block-atoms', '2500', '--workdir', work, '--keep-workdir')
    first = result.read(out)
    r = remolprobity(L_L10, '-o', out, '--block-atoms', '2500', '--workdir', work, '--resume')
    assert 'done before' in r.stderr
    assert compare.differences(result.comparable(first), result.comparable(result.read(out))) == []


@pytest.mark.container
def test_stock_reduce_refuses_blocks_but_runs_one(tmp_path):
    out = str(tmp_path / 'r.pkl')
    r = remolprobity(L_L10, '-o', out, '--block-atoms', '2500', '--reduce', STOCK_REDUCE, check=False)
    assert r.returncode == 1 and 'self-check' in r.stderr
    # refused also when the default block size would give one block: only --block-atoms 0 goes on
    r = remolprobity(L_L10, '-o', out, '--reduce', STOCK_REDUCE, check=False)
    assert r.returncode == 1 and 'self-check' in r.stderr and not os.path.exists(out)
    remolprobity(L_L10, '-o', out, '--block-atoms', '0', '--reduce', STOCK_REDUCE)
    assert result.read(out)['provenance']['reduce_self_check']['passed'] is False


@pytest.mark.container
def test_zero_occupancy_option_is_recorded(tmp_path):
    out = str(tmp_path / 'r.pkl')
    remolprobity(os.path.join(DATA, '1crn.pdb'), '-o', out, '--zero-occupancy-as-one')
    assert result.read(out)['provenance']['options']['zero_occupancy_as_one'] is True


@pytest.mark.container
def test_block_files_keep_the_models_chain_breaks(tmp_path):
    # Review Focus 3: each block file's chain objects partition its residues as the model's runs do
    import numpy as np
    from remolprobity import blocks, io
    p = os.path.join(DATA, '9a1n_split_chain.cif')
    m = io.read_model(p)
    plan = blocks.plan_blocks(m, 2500, 20.0)
    out = tmp_path / 'b'
    out.mkdir()
    paths = io.write_blocks(m, plan.members, str(out))
    script = tmp_path / 'chains.py'
    script.write_text(
        'import json, sys, iotbx.pdb\n'
        'r = []\n'
        'for p in sys.argv[1:]:\n'
        '    h = iotbx.pdb.input(file_name=p).construct_hierarchy()\n'
        '    r.append([[[c.id, rg.resseq.strip() + rg.icode.strip()] for rg in c.residue_groups()]\n'
        '              for c in h.models()[0].chains()])\n'
        'print(json.dumps(r))\n')
    chains = json.loads(in_image(str(script), *paths).stdout)
    assert len(chains) == plan.n_blocks > 2
    for members, found in zip(plan.members, chains):
        want = {}
        for k in sorted(int(r) for r in members):
            want.setdefault(int(m.run_of[k]), []).append(list(m.residues[k]))
        assert found == list(want.values())


STRETCH_DRIVER = """
import sys
import iotbx.pdb
import numpy as np
from scipy.spatial import cKDTree
from remolprobity import io, naming, tools, clashes
path, use_stretch = sys.argv[1], sys.argv[2] == 'stretch'
m = io.read_model(path)
stretch = {k: int(v) for k, v in zip(m.residues, io.stretches(m))} if use_stretch else None
conv = naming.conversion(m.chain_ids, m.residue_names)
owned = lambda c, r: c == 'L' and 298 <= int(r) <= 302
h = iotbx.pdb.input(file_name=path).construct_hierarchy()
h.atoms().reset_i_seq()
t = tools.find_tools()
import tempfile
for name, sel in (('whole', None), ('block', 'chain L')):
    hh = h if sel is None else h.select(h.atom_selection_cache().selection('chain L and resseq 296:304'), copy_atoms=True)
    hh = hh.deep_copy()
    core = [a.xyz for a in hh.atoms() if owned(a.parent().parent().parent().id.strip(), a.parent().parent().resseq.strip())]
    got = clashes.block_clashes(hh, conv, owned, cKDTree(core), 100.0, t, tempfile.mkdtemp(), m.occupancy_max, stretch)
    print(name, got['probe_atoms'])
"""


@pytest.mark.container
def test_reduce_sees_the_models_stretches(tmp_path):
    # L300 and L301 are neighbours in the block (L10 between them is dropped), not in the model:
    # reduce must not give L301 an N-terminal H neighbour it does not have in the whole model
    script = tmp_path / 'stretch.py'
    script.write_text(STRETCH_DRIVER)
    p = os.path.join(DATA, '9a1n_split_chain.cif')

    def counts(mode):
        out = in_image(str(script), p, mode).stdout.split()
        return {out[i]: int(out[i + 1]) for i in range(0, len(out), 2)}
    with_stretch = counts('stretch')
    assert with_stretch['block'] == with_stretch['whole']
    assert counts('plain')['block'] != with_stretch['whole']


@pytest.mark.container
def test_ligand_and_water_runs_agree_with_one_block(tmp_path):
    # RCSB layout: a chain's polymer, its HEM, then its HOH (4HHB); blocks of
    # 500 atoms put ligand and water runs of one chain next to each other
    p = os.path.join(DATA, '4hhb.cif')
    one, many = str(tmp_path / 'one.pkl'), str(tmp_path / 'many.pkl')
    remolprobity(p, '-o', one, '--block-atoms', '0')
    remolprobity(p, '-o', many, '--block-atoms', '500', '--margin', '21', '--nproc', '4')
    assert result.read(many)['provenance']['blocks']['count'] > 2
    assert compare.differences(result.comparable(result.read(one)), result.comparable(result.read(many)),
                               tol=1e-9) == []


@pytest.mark.container
def test_user_workdir_keeps_other_files(tmp_path):
    p, out = os.path.join(DATA, '1ehz_10nt.cif'), str(tmp_path / 'r.pkl')
    work = tmp_path / 'userdir'
    (work / 'blocks').mkdir(parents=True)
    (work / 'notes.txt').write_text('mine')
    r = remolprobity(p, '-o', out, '--workdir', str(work), check=False)
    assert r.returncode == 1 and 'not empty' in r.stderr
    assert (work / 'notes.txt').read_text() == 'mine' and (work / 'blocks').is_dir()
    assert not os.path.exists(out)
    # an empty directory is used and left empty
    empty = tmp_path / 'empty'
    empty.mkdir()
    remolprobity(p, '-o', out, '--workdir', str(empty))
    assert os.listdir(empty) == []
    # a directory the run made itself goes entirely
    made = tmp_path / 'made'
    remolprobity(p, '-o', out, '--workdir', str(made))
    assert not made.exists()


@pytest.mark.container
def test_resume_refuses_other_tools(tmp_path):
    work, out = str(tmp_path / 'work'), str(tmp_path / 'r.pkl')
    p = os.path.join(DATA, '1ehz_10nt.cif')
    remolprobity(p, '-o', out, '--workdir', work, '--keep-workdir', '--block-atoms', '0')
    r = remolprobity(p, '-o', out, '--workdir', work, '--resume', '--block-atoms', '0',
                     '--reduce', STOCK_REDUCE, check=False)
    assert r.returncode == 1 and 'changed' in r.stderr


@pytest.mark.container
def test_negative_block_atoms_refused(tmp_path):
    r = remolprobity(os.path.join(DATA, '1ehz_10nt.cif'), '-o', str(tmp_path / 'r.pkl'),
                     '--block-atoms', '-5', check=False)
    assert r.returncode == 1 and '--block-atoms' in r.stderr


def _cif_with_value(src, out, column, value, every=False):
    """src with the first atom's value (or every atom's) in one atom_site column replaced"""
    from remolprobity import io as rio
    tags = next(rio.cif_rows(src))
    lines, done = [], False
    for line in open(src):
        if not done and line.startswith(('ATOM', 'HETATM')):
            row = rio.cif_tokens(line)
            row[tags.index(column)] = value
            line = ' '.join(rio.cif_value(t) for t in row) + '\n'
            done = not every
        lines.append(line)
    open(out, 'w').write(''.join(lines))


@pytest.mark.container
def test_unreadable_models_are_refused_before_planning(tmp_path):
    # what stock MolProbity refuses too: '.' for B (9A1L)
    out = str(tmp_path / 'r.pkl')
    cif = str(tmp_path / 'b.cif')
    _cif_with_value(os.path.join(DATA, '1ehz_10nt.cif'), cif, 'B_iso_or_equiv', '.')
    r = remolprobity(cif, '-o', out, '--block-atoms', '0', check=False)
    assert r.returncode == 1 and "'.' in _atom_site.B_iso_or_equiv" in r.stderr and 'error:' in r.stderr
    assert 'Traceback' not in r.stderr and 'reading' in r.stderr and 'blocks' not in r.stderr
    assert not os.path.exists(out)


@pytest.mark.container
def test_missing_occupancy_runs_with_zero_occupancy_as_one(tmp_path):
    out = str(tmp_path / 'r.pkl')
    cif = str(tmp_path / 'o.cif')
    _cif_with_value(os.path.join(DATA, '1ehz_10nt.cif'), cif, 'occupancy', '?')
    r = remolprobity(cif, '-o', out, check=False)
    assert r.returncode == 1 and "'?' in _atom_site.occupancy" in r.stderr and 'Traceback' not in r.stderr
    remolprobity(cif, '-o', out, '--zero-occupancy-as-one')
    assert result.read(out)['provenance']['options']['zero_occupancy_as_one'] is True


@pytest.mark.container
def test_missing_b_runs_with_missing_b_as_zero_and_b_changes_nothing(tmp_path):
    # no validation here uses B: 4HHB with its B values and with none at all (read as 0) agree exactly
    src = os.path.join(DATA, '4hhb.cif')
    cif = str(tmp_path / 'no_b.cif')
    _cif_with_value(src, cif, 'B_iso_or_equiv', '.', every=True)
    r = remolprobity(cif, '-o', str(tmp_path / 'refused.pkl'), check=False)
    assert r.returncode == 1 and 'or use --missing-b-as-zero' in r.stderr and 'Traceback' not in r.stderr
    remolprobity(cif, '-o', str(tmp_path / 'no_b.pkl'), '--missing-b-as-zero', '--geometry', 'v34,v33')
    remolprobity(src, '-o', str(tmp_path / 'b.pkl'), '--geometry', 'v34,v33')
    with_b, without = (result.read(str(tmp_path / n)) for n in ('b.pkl', 'no_b.pkl'))
    assert without['provenance']['options'] == {'zero_occupancy_as_one': False, 'missing_b_as_zero': True}
    assert with_b['provenance']['options'] == {'zero_occupancy_as_one': False, 'missing_b_as_zero': False}
    assert compare.differences(result.comparable(with_b), result.comparable(without), tol=0) == []


@pytest.mark.container
def test_blocks_needing_a_larger_margin_run_again_with_it(tmp_path):
    # at 15 Å, L/L10's reduce cliques reach past 2,500-atom blocks' margins (they ask for 18)
    one, many = str(tmp_path / 'one.pkl'), str(tmp_path / 'many.pkl')
    remolprobity(L_L10, '-o', one, '--block-atoms', '0', '--geometry', 'v34,v33')
    r = remolprobity(L_L10, '-o', many, '--block-atoms', '2500', '--margin', '15', '--nproc', '4',
                     '--geometry', 'v34,v33')
    b = result.read(many)
    retried = b['provenance']['blocks']['retried']
    # some blocks, not all: a retry of every block would hide a broken check
    assert retried and len(retried) < b['provenance']['blocks']['count'], retried
    assert all(15 < m <= 30 for m in retried.values()), retried
    assert 'running it again' in r.stderr and 'GB' in r.stderr
    assert compare.differences(result.comparable(result.read(one)), result.comparable(b), tol=1e-9) == []
    # with no B at all, read as 0, the blocks that run again get it filled in too: the same result
    no_b, out = str(tmp_path / 'no_b.cif'), str(tmp_path / 'no_b.pkl')
    _cif_with_value(L_L10, no_b, 'B_iso_or_equiv', '.', every=True)
    remolprobity(no_b, '-o', out, '--block-atoms', '2500', '--margin', '15', '--nproc', '4', '--geometry', 'v34,v33',
                 '--missing-b-as-zero')
    assert result.read(out)['provenance']['blocks']['retried'] == retried
    assert compare.differences(result.comparable(b), result.comparable(result.read(out)), tol=0) == []


@pytest.mark.container
def test_resume_after_blocks_ran_again(tmp_path):
    work, out = str(tmp_path / 'work'), str(tmp_path / 'r.pkl')
    args = ('-o', out, '--block-atoms', '2500', '--margin', '15', '--nproc', '4', '--workdir', work)
    remolprobity(L_L10, *args, '--keep-workdir')
    first = result.read(out)
    assert first['provenance']['blocks']['retried']
    r = remolprobity(L_L10, *args, '--resume')
    # the first round and the larger-margin round both come from their stores
    assert r.stderr.count('done before') == 2 and ' done (' not in r.stderr, r.stderr[-3000:]
    again = result.read(out)
    assert again['provenance']['blocks']['retried'] == first['provenance']['blocks']['retried']
    assert compare.differences(result.comparable(first), result.comparable(again)) == []


@pytest.mark.container
def test_a_block_needing_more_than_twice_the_margin_stops_the_run(tmp_path):
    # 4HHB in 300-atom blocks needs 21 Å: more than twice the minimum margin of 10
    out = str(tmp_path / 'r.pkl')
    r = remolprobity(os.path.join(DATA, '4hhb.cif'), '-o', out, '--block-atoms', '300', '--margin', '10',
                     '--nproc', '8', check=False)
    assert r.returncode == 1 and not os.path.exists(out)
    assert 'MarginError: ' in r.stderr and 'or more' in r.stderr and 'up to twice --margin, 20 Å' in r.stderr
    assert 'Traceback' not in r.stderr


@pytest.mark.container
def test_a_known_refusal_in_a_block_has_no_traceback(tmp_path):
    # Review Focus 5 through the program: duplicate atoms are found in the
    # block's worker; the user sees the block and the message, and the
    # traceback stays in the workdir
    src = open(os.path.join(DATA, '1crn.pdb')).read().splitlines(True)
    first = next(i for i, l in enumerate(src) if l.startswith('ATOM'))
    p = tmp_path / 'dup.pdb'
    p.write_text(''.join(src[:first + 1] + [src[first]] + src[first + 1:]))
    out, work = str(tmp_path / 'r.pkl'), tmp_path / 'work'
    r = remolprobity(str(p), '-o', out, '--workdir', str(work), check=False)
    assert r.returncode == 1 and not os.path.exists(out)
    assert 'error: block 0: duplicate atoms (same residue, name and altloc)' in r.stderr, r.stderr[-3000:]
    assert 'Traceback' not in r.stderr
    kept = work / 'results' / 'block00000.traceback.txt'
    assert 'traceback in %s' % kept in r.stderr and 'Traceback' in kept.read_text()


CLIQUES = """
import json, sys
import iotbx.pdb
from remolprobity import clashes, io, naming, tools
from remolprobity.ids import residue_id
path, work = sys.argv[1], sys.argv[2]
m = io.read_model(path)
h = iotbx.pdb.input(file_name=path).construct_hierarchy()
hp, names = naming.pdb_copy(h, naming.conversion(m.chain_ids, m.residue_names))
noh = work + '/noh.pdb'
with open(noh, 'w') as f:
    f.write(hp.as_pdb_string())
withh = tools.add_hydrogens(tools.find_tools(), noh, work)
residues = {(l[20:22].strip(), residue_id(l[22:26], l[26])) for l in open(withh) if l.startswith(('ATOM', 'HETATM'))}
members = [x for c in clashes._cliques(withh) for x in c]
print(json.dumps({'members': len(members), 'found': sum(tuple(x) in residues for x in members),
                  'above_9999': sum(int(x[1].rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')) > 9999 for x in members)}))
"""


@pytest.mark.container
def test_cliques_above_residue_9999(tmp_path):
    # reduce writes the residue numbers of the groups it flips (amide, His) in
    # decimal and of those it rotates in hybrid-36: past 9999 they differ, and
    # every clique member must still be found, or its clique goes unchecked
    import sys
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    import make_fixtures
    half, renumbered = str(tmp_path / 'half.cif'), str(tmp_path / 'renumbered.cif')
    make_fixtures.renumber(L_L10, half, 'L', 10000)
    make_fixtures.renumber(half, renumbered, 'L10', 10000)

    def cliques(path, name):
        work = tmp_path / name
        work.mkdir()
        return json.loads(in_image('-c', CLIQUES, path, str(work)).stdout.strip().splitlines()[-1])
    before, after = cliques(L_L10, 'before'), cliques(renumbered, 'after')
    assert before['members'] == before['found'] > 20 and before['above_9999'] == 0
    assert after == {'members': before['members'], 'found': before['members'], 'above_9999': before['members']}
    one, many = str(tmp_path / 'one.pkl'), str(tmp_path / 'many.pkl')
    remolprobity(renumbered, '-o', one, '--block-atoms', '0', '--geometry', 'v34,v33')
    remolprobity(renumbered, '-o', many, '--block-atoms', '2500', '--margin', '20', '--nproc', '4',
                 '--geometry', 'v34,v33')
    a, b = result.read(one), result.read(many)
    assert b['provenance']['blocks']['count'] > 2
    assert all(int(r['resid']) > 9999 for r in a['residues'])
    assert compare.differences(result.comparable(a), result.comparable(b), tol=1e-9) == []


@pytest.mark.container
def test_output_inside_the_workdir_is_refused(tmp_path):
    # a successful run removes its workdir, or what it made there: the result must not be in it
    p = os.path.join(DATA, '1ehz_10nt.cif')
    new, empty = tmp_path / 'new', tmp_path / 'empty'
    empty.mkdir()
    for work, out in ((new, new / 'r.pkl'), (empty, empty / 'sub' / 'r.pkl'),
                      (empty, tmp_path / 'other' / '..' / 'empty' / 'r.pkl'), (empty, empty)):
        r = remolprobity(p, '-o', str(out), '--workdir', str(work), check=False)
        assert r.returncode == 1 and 'inside --workdir' in r.stderr, r.stderr[-2000:]
        # before any work
        assert 'Traceback' not in r.stderr and 'reading' not in r.stderr
    assert not new.exists() and os.listdir(empty) == []


@pytest.mark.container
def test_a_refusal_before_the_plan_leaves_the_workdir_as_found(tmp_path):
    # stock reduce fails the self-check once the workdir holds its scratch: nothing to resume, so the
    # run takes back what it made, and the same directory can be used again
    p, out = os.path.join(DATA, '1ehz_10nt.cif'), str(tmp_path / 'r.pkl')
    empty, made = tmp_path / 'empty', tmp_path / 'made'
    empty.mkdir()
    for work in (empty, made, None):
        args = ['--workdir', str(work)] if work else []
        r = remolprobity(p, '-o', out, '--block-atoms', '100', '--reduce', STOCK_REDUCE, *args, check=False)
        assert r.returncode == 1 and 'self-check' in r.stderr, r.stderr[-2000:]
    assert os.listdir(empty) == [] and not made.exists()
    assert sorted(os.listdir(tmp_path)) == ['empty']  # no default workdir left beside the output
    remolprobity(p, '-o', out, '--block-atoms', '100', '--workdir', str(empty))
    assert result.read(out)['provenance']['blocks']['count'] > 1 and os.listdir(empty) == []
