import json
import os

import pytest

from helpers import DATA, in_image

DRIVER = """
import json, sys
import iotbx.pdb
from iotbx.data_manager import DataManager
from libtbx.utils import null_out
from mmtbx.validation.mp_validate_bonds import mp_validate_bonds
from mmtbx.validation.restraints import combined
from remolprobity import geometry, io
from remolprobity.ids import atom_tuple
path = sys.argv[1]
m = io.read_model(path)
h = iotbx.pdb.input(file_name=path).construct_hierarchy()
whole = {d: geometry.score(d, h.deep_copy(), lambda c, r: True) for d in geometry.DEFINITIONS}
half = set(m.residues[:len(m.residues) // 2])
parts = [{d: geometry.score(d, h.deep_copy(), f) for d in geometry.DEFINITIONS}
         for f in (lambda c, r: (c, r) in half, lambda c, r: (c, r) not in half)]
# stock, as IHMValidation 3.4's molprobity_convert.py scores bonds
dm = DataManager(['model']); dm.process_model_file(path); model = dm.get_model()
model.set_stop_for_unknowns(False)
p = model.get_default_pdb_interpretation_params()
p.pdb_interpretation.allow_polymer_cross_special_position = True
p.pdb_interpretation.flip_symmetric_amino_acids = False
p.pdb_interpretation.clash_guard.nonbonded_distance_threshold = None
p.pdb_interpretation.restraints_library.cdl = True
model.set_log(null_out()); model.process(make_restraints=True, pdb_interpretation_params=p)
r = mp_validate_bonds(pdb_hierarchy=model.get_hierarchy(),
                      geometry_restraints_manager=model.get_restraints_manager().geometry, outliers_only=False)
noh = lambda x: not any(a.element.strip().upper() in ('H', 'D') for a in x.atoms_info)
out = {d: {k: [whole[d][k]['total'], sum(x[d][k]['total'] for x in parts)] for k in ('bonds', 'angles')}
       for d in geometry.DEFINITIONS}
out['stock_v34'] = [sum(1 for x in r.bonds.results if noh(x)), sum(1 for x in r.angles.results if noh(x))]
out['residues_split'] = len(whole['v34']['residues']) == sum(len(x['v34']['residues']) for x in parts)

def close(a, b):
    return len(a) == len(b) and all(x[:-4] == y[:-4] and all(abs(u - v) < 1e-9 for u, v in zip(x[-4:], y[-4:]))
                                    for x, y in zip(sorted(a), sorted(b)))
at = lambda x: tuple(atom_tuple(a) for a in x.atoms_info)
stock34 = {'bonds': sorted(at(x) + (float(x.distance_value), float(x.target), float(x.sigma), float(x.score))
                           for x in r.bonds.results if noh(x) and abs(x.score) > geometry.V34_CUTOFF),
           'angles': sorted(at(x) + (float(x.angle_value), float(x.target), float(x.sigma), float(x.score))
                            for x in r.angles.results if noh(x) and abs(x.score) > geometry.V34_CUTOFF)}
out['v34_outliers'] = [[len(whole['v34'][k]['outliers']), len(stock34[k]), close(whole['v34'][k]['outliers'], stock34[k])]
                       for k in ('bonds', 'angles')]
# stock v33: restraints.combined as molprobity.molprobity runs it (no CDL)
m33 = dm.get_model()
m33.set_stop_for_unknowns(False)
q = m33.get_default_pdb_interpretation_params()
q.pdb_interpretation.clash_guard.nonbonded_distance_threshold = None
q.pdb_interpretation.restraints_library.cdl = False
m33.set_log(null_out()); m33.process(make_restraints=True, pdb_interpretation_params=q)
c = combined(pdb_hierarchy=m33.get_hierarchy(), xray_structure=m33.get_xray_structure(),
             geometry_restraints_manager=m33.get_restraints_manager().geometry, ignore_hd=True,
             reverse_sort=True, cdl=None)
stock33 = {k: sorted(at(x) + (float(x.model), float(x.target), float(x.score)) for x in getattr(c, k).results)
           for k in ('bonds', 'angles')}
out['v33_outliers'] = []
for k in ('bonds', 'angles'):
    mine = sorted(whole['v33'][k]['outliers'])
    out['v33_outliers'].append([len(mine), len(stock33[k]), len(mine) == len(stock33[k]) and all(
        x[:-3] == y[:-3] and all(abs(u - v) < 1e-9 for u, v in zip(x[-3:], y[-3:])) for x, y in zip(mine, stock33[k]))])
print(json.dumps(out))
"""


@pytest.mark.container
def test_totals_match_stock_and_split_by_ownership():
    out = json.loads(in_image('-c', DRIVER, os.path.join(DATA, '1crn.pdb')).stdout.strip().splitlines()[-1])
    assert out['v34']['bonds'][0] == out['v34']['bonds'][1] == out['stock_v34'][0]
    assert out['v34']['angles'][0] == out['v34']['angles'][1] == out['stock_v34'][1]
    assert out['v33']['bonds'][0] == out['v33']['bonds'][1] > 0
    assert out['residues_split']
    # 1CRN has 1+3 v34 and 1+6 v33 outliers; identities exact, values within 1e-9
    assert [x[:2] for x in out['v34_outliers']] == [[1, 1], [3, 3]]
    assert [x[:2] for x in out['v33_outliers']] == [[1, 1], [6, 6]]
    assert all(x[2] for x in out['v34_outliers'] + out['v33_outliers'])


GLYCAN = """
import json, sys
import iotbx.pdb
from iotbx.data_manager import DataManager
from libtbx.utils import null_out
try:
    from mmtbx.monomer_library import tst_glyco_coordination as t
    text = t.pdbs['linking_BMA.pdb']
except (ImportError, KeyError) as e:
    print(json.dumps({'skip': repr(e)})); sys.exit(0)
path = sys.argv[1]
open(path, 'w').write(text)
from mmtbx.validation.mp_validate_bonds import mp_validate_bonds
from remolprobity import geometry
from remolprobity.ids import atom_tuple
h = iotbx.pdb.input(file_name=path).construct_hierarchy()
mine = geometry.score('v34', h.deep_copy(), lambda c, r: True)
dm = DataManager(['model']); dm.process_model_file(path); model = dm.get_model()
model.set_stop_for_unknowns(False)
p = model.get_default_pdb_interpretation_params()
p.pdb_interpretation.allow_polymer_cross_special_position = True
p.pdb_interpretation.flip_symmetric_amino_acids = False
p.pdb_interpretation.clash_guard.nonbonded_distance_threshold = None
p.pdb_interpretation.restraints_library.cdl = True
model.set_log(null_out()); model.process(make_restraints=True, pdb_interpretation_params=p)
r = mp_validate_bonds(pdb_hierarchy=model.get_hierarchy(),
                      geometry_restraints_manager=model.get_restraints_manager().geometry, outliers_only=False)
noh = lambda x: not any(a.element.strip().upper() in ('H', 'D') for a in x.atoms_info)
at = lambda x: tuple(atom_tuple(a) for a in x.atoms_info)
out = {}
for k, val in (('bonds', 'distance_value'), ('angles', 'angle_value')):
    res = [x for x in getattr(r, k).results if noh(x)]
    st = sorted(at(x) + (float(getattr(x, val)), float(x.target), float(x.sigma), float(x.score))
                for x in res if abs(x.score) > geometry.V34_CUTOFF)
    me = sorted(mine[k]['outliers'])
    out[k] = [mine[k]['total'], len(res), len(me), len(st),
              len(me) == len(st) and all(a[:-4] == b[:-4] and all(abs(u - v) < 1e-9 for u, v in zip(a[-4:], b[-4:]))
                                         for a, b in zip(me, st))]
try:  # as geometry.py finds it: cctbx from 2026-08-19 scores covalent geometry and SS BOND only
    from mmtbx.validation.mp_validate_bonds import _scoreable_origin_ids
    out['scoreable'] = True
except ImportError:
    out['scoreable'] = False
print(json.dumps(out))
"""


@pytest.mark.container
def test_glycan_link_restraints_are_not_scored(tmp_path):
    out = json.loads(in_image('-c', GLYCAN, str(tmp_path / 'linking_BMA.pdb')).stdout.strip().splitlines()[-1])
    if 'skip' in out:
        pytest.skip('linking_BMA.pdb not available: %s' % out['skip'])
    for kind in ('bonds', 'angles'):
        mine_total, stock_total, n_mine, n_stock, same = out[kind]
        assert mine_total == stock_total
        assert n_mine == n_stock and same
    # cctbx from 2026-08-19 leaves the glycan links out of the 65 bonds older
    # cctbx scores; either way remolprobity scores what stock does (above)
    assert out['bonds'][1] == (61 if out['scoreable'] else 65)
