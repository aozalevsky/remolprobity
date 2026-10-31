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
