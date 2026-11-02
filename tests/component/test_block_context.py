"""A block's chains as cctbx sees them in the whole model: conformer order, links across left-out residues"""
import json
import os

import pytest

from helpers import DATA, in_image

PREP = """
import sys
import iotbx.pdb
src, out = sys.argv[1], sys.argv[2]
h = iotbx.pdb.input(file_name=src).construct_hierarchy()
# residue 1: altlocs A then B, so the chain's conformers are A, B; residue 19: B (moved) then A
for rg in h.models()[0].chains()[0].residue_groups():
    n = int(rg.resseq)
    if n in (1, 19):
        ag = rg.atom_groups()[0]
        other = ag.detached_copy()
        for a in ag.atoms():
            a.occ = 0.5
        for a in other.atoms():
            a.occ = 0.5
            if n == 19:
                a.xyz = (a.xyz[0] + 0.6, a.xyz[1] - 0.4, a.xyz[2] + 0.3)
        first, second = (ag, other) if n == 1 else (other, ag)
        first.altloc, second.altloc = ('A', 'B') if n == 1 else ('B', 'A')
        rg.remove_atom_group(ag)
        rg.append_atom_group(first)
        rg.append_atom_group(second)
open(out, 'w').write(h.as_pdb_string())
"""

DRIVER = """
import json, sys, tempfile
import numpy as np
import iotbx.pdb
from remolprobity import blocks, io, residues
path, context = sys.argv[1], sys.argv[2] == '1'
m = io.read_model(path)
owner = np.array([0 if r in ('19', '20') else 1 for _, r in m.residues])  # residue 1 is 12 Å from both
if not context:
    m.introducers = {}
members = blocks.block_members(m, owner, 0, 10.0)
block = io.write_blocks(m, [members], tempfile.mkdtemp())[0]

def view(p):
    h = iotbx.pdb.input(file_name=p).construct_hierarchy()
    order = [c.altloc for c in h.models()[0].chains()[0].conformers()]
    rama = sorted(r for r in residues.ramachandran(h) if r[1] in ('19', '20', '21'))
    return order, rama

whole, part = view(path), view(block)
print(json.dumps({'whole': whole[0], 'block': part[0], 'has_1': 0 in set(members.tolist()),
                  'same_20': whole[1] == part[1], 'rama': [whole[1], part[1]]}))
"""


@pytest.mark.container
@pytest.mark.parametrize('context', [True, False], ids=['with_context', 'without'])
def test_block_keeps_the_conformer_order(tmp_path, context):
    # iotbx orders a chain's conformers by the first appearance of each altloc. ramalyze and omegalyze label
    # a result by the altlocs of its own atoms, so their results here do not depend on the order; the
    # conformation-dependent library's averaging over three or more conformers does
    src = tmp_path / 'alt.pdb'
    in_image('-c', PREP, os.path.join(DATA, '1crn.pdb'), src)
    out = json.loads(in_image('-c', DRIVER, src, '1' if context else '0').stdout.strip().splitlines()[-1])
    assert out['whole'] == ['A', 'B']
    if context:
        assert out['has_1'] and out['block'] == ['A', 'B'] and out['same_20'], out
    else:
        # the case the context is for: without residue 1, the block's first conformer is B
        assert not out['has_1'] and out['block'] == ['B', 'A'], out


SLICE_DRIVER = """
import json, os, sys, tempfile
import iotbx.pdb
from scitbx.array_family import flex
from remolprobity import blocks, io

data_dir = sys.argv[1]
cases = [
    (os.path.join(data_dir, '1crn.pdb'), 100, 10.0, False, False, False),
    (os.path.join(data_dir, '1ehz_10nt.cif'), 100, 10.0, False, False, False),
    (os.path.join(data_dir, '4hhb.cif'), 1500, 25.0, True, True, True),
    (os.path.join(data_dir, '9a1n_split_chain.cif'), 5000, 25.0, False, False, False),
    (os.path.join(data_dir, '9a1n_L_L10.cif'), 5000, 25.0, True, True, False),
]
checked = 0
for src, block_atoms, margin, z_occ, m_b, add_utf8 in cases:
    tmp = tempfile.mkdtemp()
    if add_utf8:
        raw = open(src, 'r', encoding='utf-8').read()
        raw = raw.replace('data_', "data_utf8\\n_struct.title 'Hämoglobin, deoxy — α₂β₂'\\n#\\ndata_", 1).split('#\\ndata_', 1)[0] + open(src, 'r', encoding='utf-8').read().split('\\n', 1)[1]
        p = os.path.join(tmp, 'utf8.cif')
        open(p, 'w', encoding='utf-8').write(raw)
        src = p
    m = io.read_model(src, zero_occupancy_as_one=z_occ, missing_b_as_zero=m_b)
    assert m.spans is not None, src
    plan = blocks.plan_blocks(m, block_atoms, margin)
    bdir = os.path.join(tmp, 'blocks')
    os.mkdir(bdir)
    paths = io.write_blocks(m, plan.members, bdir, zero_occupancy_as_one=z_occ, missing_b_as_zero=m_b)
    io.check_segments(m, plan.members, paths)
    ms = io.model_slice(m, zero_occupancy_as_one=z_occ, missing_b_as_zero=m_b)
    for b, (mem, bpath) in enumerate(zip(plan.members, paths)):
        txt = io.block_text(ms, mem, path=bpath)
        h_disk = iotbx.pdb.input(file_name=bpath).construct_hierarchy(sort_atoms=False)
        h_mem = iotbx.pdb.input(source_info=None, lines=flex.split_lines(txt)).construct_hierarchy(sort_atoms=False)
        assert [a.id_str() for a in h_disk.atoms()] == [a.id_str() for a in h_mem.atoms()], (src, b)
        assert list(h_disk.atoms().extract_xyz()) == list(h_mem.atoms().extract_xyz()), (src, b)
        assert list(h_disk.atoms().extract_occ()) == list(h_mem.atoms().extract_occ()), (src, b)
        assert list(h_disk.atoms().extract_b()) == list(h_mem.atoms().extract_b()), (src, b)
        checked += 1
print(json.dumps({'checked': checked}))
"""


@pytest.mark.container
def test_in_memory_block_hierarchies_match_written_blocks():
    out = json.loads(in_image('-c', SLICE_DRIVER, DATA).stdout.strip().splitlines()[-1])
    assert out['checked'] >= 10

