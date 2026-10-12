import json
import os

import pytest

from helpers import DATA, REPO, in_image

DRIVER = """
import json, sys
import iotbx.pdb
from mmtbx.validation.ramalyze import ramalyze
from libtbx.utils import null_out
from remolprobity import io, naming, residues
from remolprobity.ids import residue_id
m = io.read_model(sys.argv[1])
h = iotbx.pdb.input(file_name=sys.argv[1]).construct_hierarchy()
conv = naming.conversion(m.chain_ids, m.residue_names)
hp, names = naming.pdb_copy(h, conv)
keys = sorted(set((c, r) for c, r, _ in names.values()))
rama = residues.ramachandran(h)
whole = ramalyze(pdb_hierarchy=h, outliers_only=False, out=null_out(), quiet=True)
whole = [(x.chain_id.strip(), residue_id(x.resseq, x.icode), x.resname.strip(), x.altloc.strip(),
          x.ramalyze_type(), float(x.phi), float(x.psi), float(x.score)) for x in whole.results]
rota = residues.rotamers(h)
print(json.dumps({'converted': conv is not None, 'fits': hp.fits_in_pdb_format(),
                  'pdb_chains': sorted(set(c.id for c in hp.chains())),
                  'keys_match': keys == sorted(m.residues), 'rama_same': sorted(rama) == sorted(whole),
                  'rama': len(rama), 'rota': len(rota)}))
"""


@pytest.mark.container
def test_names_and_residue_records():
    out = json.loads(in_image('-c', DRIVER, os.path.join(DATA, '9a1n_L_L10.cif')).stdout.strip().splitlines()[-1])
    assert out['converted'] and out['fits']
    assert all(len(c.strip()) <= 2 for c in out['pdb_chains'])
    assert out['keys_match']
    assert out['rama_same'] and out['rama'] == 1230 and out['rota'] == 1158


# chain ids of 1, 3 and 4 characters, and a 4-character residue name
LONG = """data_long
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . GLY 1 1 1 ? A A 0 0 0 1 10 1
ATOM 2 C CA . GLY 1 1 1 ? B L10 5 0 0 1 10 1
ATOM 3 C CA . GLY 1 1 1 ? C WXYZ 10 0 0 1 10 1
HETATM 4 C C1 . XYZW 1 . 2 ? D WXYZ 15 0 0 1 10 1
#
"""

LONG_DRIVER = """
import json, sys
import iotbx.pdb
from remolprobity import io, naming
m = io.read_model(sys.argv[1])
h = iotbx.pdb.input(file_name=sys.argv[1]).construct_hierarchy()
hp, names = naming.pdb_copy(h, naming.conversion(m.chain_ids, m.residue_names))
print(json.dumps({'fits': hp.fits_in_pdb_format(), 'pdb_chains': [c.id for c in hp.chains()],
                  'pdb_resnames': [ag.resname for ag in hp.atom_groups()],
                  'back': sorted([list(v) for v in names.values()])}))
"""


@pytest.mark.container
def test_long_chain_ids_and_residue_names(tmp_path):
    p = tmp_path / 'long.cif'
    p.write_text(LONG)
    out = json.loads(in_image('-c', LONG_DRIVER, str(p)).stdout.strip().splitlines()[-1])
    assert out['fits']
    assert all(len(c) <= 2 for c in out['pdb_chains']) and all(len(n) == 3 for n in out['pdb_resnames'])
    assert out['back'] == [['A', '1', 'GLY'], ['L10', '1', 'GLY'], ['WXYZ', '1', 'GLY'], ['WXYZ', '2', 'XYZW']]


@pytest.mark.container
def test_ramachandran_per_chain_on_a_split_chain(tmp_path):
    import sys

    from remolprobity import io

    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    import make_fixtures

    p = str(tmp_path / 'split.cif')
    make_fixtures.split_chain(os.path.join(DATA, '9a1n_L_L10.cif'), p, 'L', 300)
    order = [c for c, _ in io.read_model(p).residues]
    assert [c for i, c in enumerate(order) if i == 0 or c != order[i - 1]] == ['L', 'L10', 'L']
    out = json.loads(in_image('-c', DRIVER, p).stdout.strip().splitlines()[-1])
    assert out['keys_match'] and out['rama_same']
    # 1230 unsplit. The two L segments are separate chain objects in cctbx, so L 300 and
    # L 301 lose their phi/psi neighbours; whole-model ramalyze agrees (rama_same)
    assert out['rama'] == 1228
