import re

import pytest

from remolprobity import io

PDB = (
    "HEADER    TEST\n"
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 10.00           N\n"
    "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00 10.00           C\n"
    "ATOM      3  CA  ALA AA000A      3.000   1.000   0.000  0.00 10.00           C\n"
    "HETATM    4  H1  HOH B   5       9.000   9.000   9.000  1.00 10.00           H\n"
    "END\n")


def test_read_pdb(tmp_path):
    p = tmp_path / 'm.pdb'
    p.write_text(PDB)
    m = io.read_model(str(p))
    assert m.format == 'pdb'
    assert m.residues == [('A', '1'), ('A', '10000A'), ('B', '5')]
    assert m.resnames == ['GLY', 'ALA', 'HOH']
    assert m.zero_occupancy == 1
    assert m.hydrogens == 1
    assert m.position[1].tolist() == [3.0, 1.0, 0.0]


def test_two_models_are_refused(tmp_path):
    p = tmp_path / 'm.pdb'
    p.write_text("MODEL        1\n" + PDB.replace('END\n', '') + "ENDMDL\nMODEL        2\nENDMDL\n")
    with pytest.raises(io.InputError, match='more than one model'):
        io.read_model(str(p))


BASE = 'ATOM      1  CA  ALA A   1      11.104   6.134  -6.504  1.00 10.00           C'
FIELDS = {'x': (30, 38), 'y': (38, 46), 'z': (46, 54), 'occupancy': (54, 60), 'B': (60, 66)}


def pdb_with(tmp_path, **values):
    line = BASE
    for field, value in values.items():
        i, j = FIELDS[field]
        line = line[:i] + value.rjust(j - i) + line[j:]
    p = tmp_path / 'm.pdb'
    p.write_text(line + '\n')
    return str(p)


@pytest.mark.parametrize('field', sorted(FIELDS))
@pytest.mark.parametrize('value', ['.', '?', 'nan', '-', '1_0', 'inf'])
def test_number_iotbx_cannot_read_is_refused_naming_atom_and_columns(tmp_path, field, value):
    i, j = FIELDS[field]
    with pytest.raises(io.InputError, match=r"atom ATOM      1  CA  ALA A   1 has '%s' in columns %d-%d \(%s\)"
                       % (re.escape(value), i + 1, j, field)):
        io.read_model(pdb_with(tmp_path, **{field: value}))


def test_numbers_as_iotbx_reads_them(tmp_path):
    # iotbx: leading blanks ignored, later ones zeros, a blank field 0, exponents allowed
    m = io.read_model(pdb_with(tmp_path, x='', y='1 0', z='1e1', occupancy='', B=''))
    assert m.xyz.tolist() == [[0.0, 100.0, 10.0]] and m.zero_occupancy == 1
    assert io.pdb_number('12  ') == 1200.0 and io.pdb_number(' . ') == 0.0 and io.pdb_number('1.0e ') == 1.0
    assert io.read_model(pdb_with(tmp_path, x='.5', y='-.5', z='+1.', occupancy='1E0', B='1e-2')).n_atoms == 1


def test_unchecked_read_takes_what_iotbx_cannot(tmp_path):
    m = io.read_model(pdb_with(tmp_path, occupancy='?', B='.'), check=False)
    assert m.n_atoms == 1 and m.zero_occupancy == 1
