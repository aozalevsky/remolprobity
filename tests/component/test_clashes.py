import json
import os

import pytest

from helpers import DATA, in_image

DRIVER = """
import json, sys, tempfile
import iotbx.pdb
from scipy.spatial import cKDTree
from mmtbx.validation.clashscore import clashscore
from remolprobity import io, naming, tools, clashes
path, mode = sys.argv[1], sys.argv[2]
m = io.read_model(path)
h = iotbx.pdb.input(file_name=path).construct_hierarchy()
h.flip_symmetric_amino_acids()
t = tools.find_tools()
conv = naming.conversion(m.chain_ids, m.residue_names)
owned = lambda c, r: True
if mode == 'whole':
    tree = cKDTree(h.atoms().extract_xyz().as_numpy_array())
    got = clashes.block_clashes(h, conv, owned, tree, 15.0, t, tempfile.mkdtemp(), m.occupancy_max)
    stock = clashscore(pdb_hierarchy=h.deep_copy(), keep_hydrogens=False, fast=False, condensed_probe=False)
    print(json.dumps({'n': len(got['records']), 'stock_n': len(stock.results), 'atoms': got['probe_atoms'],
                      'score': len(got['records']) * 1000 / got['probe_atoms'],
                      'stock_score': stock.get_clashscore(),
                      'chains': sorted({a[0] for c in got['records'] for a in c[:2]}),
                      'overlaps': sorted(abs(c.overlap) for c in stock.results),
                      'ours': sorted(c[2] for c in got['records']),
                      'distances': [c[3] for c in got['records'][:50]]}))
else:
    # the core is a point far from every atom: every clique and clash is
    # beyond the margin's inner limit
    try:
        clashes.block_clashes(h, conv, owned, cKDTree([[1e4, 1e4, 1e4]]), 15.0, t, tempfile.mkdtemp(),
                              m.occupancy_max)
        print(json.dumps({'error': None}))
    except clashes.MarginError as e:
        print(json.dumps({'error': str(e)}))
"""
L_L10 = os.path.join(DATA, '9a1n_L_L10.cif')


@pytest.mark.container
def test_one_block_matches_stock_clashscore():
    # stock: 1,883 clashes on this excerpt
    out = json.loads(in_image('-c', DRIVER, L_L10, 'whole').stdout.strip().splitlines()[-1])
    assert out['n'] == out['stock_n'] > 1000
    assert out['overlaps'] == out['ours']
    assert abs(out['score'] - out['stock_score']) < 1e-9
    # the model's names, not the PDB names probe saw ('L10' has none of its own in PDB format)
    assert 'L10' in out['chains'] and set(out['chains']) <= {'L', 'L10'}
    assert all(0.5 < d < 4.0 for d in out['distances'])


@pytest.mark.container
def test_too_small_a_margin_is_an_error():
    out = json.loads(in_image('-c', DRIVER, L_L10, 'far').stdout.strip().splitlines()[-1])
    assert out['error'] and 'margin' in out['error'] and 'or more' in out['error']
