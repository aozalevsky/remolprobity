import gzip
import re

import numpy as np
import pytest

from remolprobity import io

HEAD = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.auth_comp_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
_atom_site.label_entity_id
"""

ROWS = """ATOM 1 N N . GLY 1 1 ? A A GLY 0.0 0.0 0.0 1.00 10.0 1 1
ATOM 2 C CA . GLY 1 1 ? A A GLY 1.0 0.0 0.0 1.00 10.0 1 1
ATOM 3 N N . ALA 2 2 A A A ALA 2.0 0.0 0.0 0.00 10.0
1 1
ATOM 4 C CA . ALA 2 2 A A A ALA 3.0 1.0 0.0 . 10.0 1 1
HETATM 5 O "O5'" . XYZW . 10 ? B LONG XYZW 9.0 9.0 9.0 1.00 10.0 1 2
ATOM 6 H H . GLY 1 1 ? A A GLY 0.5 0.5 0.0 1.00 10.0 1 1
#
"""


def write(tmp_path, text, name='m.cif'):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_cif_tokens_quoting():
    assert io.cif_tokens("A 'B C' \"D'E\" F' ?") == ['A', 'B C', "D'E", "F'", '?']


def test_cif_value_round_trip():
    for tok in ['ABC', 'B C', "O5'", '_x', '#1', '?', '.']:
        assert io.cif_tokens(io.cif_value(tok)) == [tok]


def test_read_cif_model(tmp_path):
    m = io.read_model(write(tmp_path, HEAD + ROWS), zero_occupancy_as_one=True)
    assert m.format == 'cif'
    assert m.n_atoms == 6
    assert m.residues == [('A', '1'), ('A', '2A'), ('LONG', '10')]
    assert m.resnames == ['GLY', 'ALA', 'XYZW']
    assert m.chain_ids == ['A', 'LONG']
    assert m.residue_names == ['GLY', 'ALA', 'XYZW']
    assert list(m.res_of) == [0, 0, 1, 1, 2, 0]
    assert m.zero_occupancy == 2
    assert m.occupancy_max == 1.0
    assert m.hydrogens == 1
    np.testing.assert_allclose(m.position, [[1, 0, 0], [3, 1, 0], [9, 9, 9]])
    assert list(m.atoms_per_residue()) == [3, 2, 1]
    assert len(m.sha256) == 64


def test_gzipped_cif(tmp_path):
    p = tmp_path / 'm.cif.gz'
    with gzip.open(p, 'wt') as f:
        f.write(HEAD + ROWS)
    assert io.read_model(str(p), zero_occupancy_as_one=True).n_atoms == 6


def test_more_than_one_model_is_refused(tmp_path):
    rows = ROWS.replace('10.0 1 1\nHETATM', '10.0 2 1\nHETATM')
    with pytest.raises(io.InputError, match='more than one model'):
        io.read_model(write(tmp_path, HEAD + rows), zero_occupancy_as_one=True)


def test_missing_residue_number_is_refused(tmp_path):
    # Review Focus 1: iotbx refuses this; so do we, naming the atom
    rows = ROWS.replace('ALA 2 2 A A A ALA 3.0', 'ALA 2 . A A A ALA 3.0')
    with pytest.raises(io.InputError, match='residue number'):
        io.read_model(write(tmp_path, HEAD + rows))


def test_reappearing_chain_keeps_one_key_per_residue(tmp_path):
    # Review Focus 3: chain A, then B, then A again
    rows = ("ATOM 1 C CA . GLY 1 1 ? A A GLY 0 0 0 1 1 1 1\n"
            "ATOM 2 C CA . GLY 1 1 ? B B GLY 5 0 0 1 1 1 1\n"
            "ATOM 3 C CA . GLY 2 2 ? A A GLY 9 0 0 1 1 1 1\n#\n")
    m = io.read_model(write(tmp_path, HEAD + rows))
    assert m.residues == [('A', '1'), ('B', '1'), ('A', '2')]
    assert m.chain_ids == ['A', 'B']


def test_row_cut_short_is_refused(tmp_path):
    with pytest.raises(io.InputError, match='middle of a row'):
        io.read_model(write(tmp_path, HEAD + "ATOM 1 N N . GLY 1 1 ? A A\n#\n"))


# ROWS with every occupancy given: ROWS has a '.' on purpose, for --zero-occupancy-as-one
CLEAN = ROWS.replace('3.0 1.0 0.0 . 10.0', '3.0 1.0 0.0 1.00 10.0')
COLUMNS = HEAD.split()[2:]


def with_values(changes):
    """CLEAN with these values replaced: {(atom, column): value}"""
    lines = CLEAN.splitlines(keepends=True)
    for (atom, column), value in changes.items():
        row = lines[atom - 1].split()
        row[COLUMNS.index('_atom_site.' + column)] = value
        lines[atom - 1] = ' '.join(row) + '\n'
    return HEAD + ''.join(lines)


def with_value(column, value, atom=2):
    """CLEAN with one atom's value in one column replaced"""
    return with_values({(atom, column): value})


@pytest.mark.parametrize('column', ['Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv'])
@pytest.mark.parametrize('value', ['.', '?'])
def test_missing_value_is_refused_naming_atom_and_column(tmp_path, column, value):
    # iotbx (so stock MolProbity) cannot read these; refused before any block is planned
    with pytest.raises(io.InputError, match=r"atom ATOM 2 C CA .*'%s'.*_atom_site\.%s" % (re.escape(value), column)):
        io.read_model(write(tmp_path, with_value(column, value)))


def test_missing_occupancy_is_read_with_zero_occupancy_as_one(tmp_path):
    # the option writes 1.00 there, so iotbx can read the blocks
    m = io.read_model(write(tmp_path, HEAD + ROWS), zero_occupancy_as_one=True)
    assert m.zero_occupancy == 2
    with pytest.raises(io.InputError, match='B_iso_or_equiv'):
        io.read_model(write(tmp_path, with_value('B_iso_or_equiv', '.')), zero_occupancy_as_one=True)


def test_missing_b_is_read_with_missing_b_as_zero(tmp_path):
    # the option writes 0.00 there, so iotbx can read the blocks
    for value in ('.', '?'):
        assert io.read_model(write(tmp_path, with_value('B_iso_or_equiv', value)), missing_b_as_zero=True).n_atoms == 6
    with pytest.raises(io.InputError, match='occupancy'):
        io.read_model(write(tmp_path, with_value('occupancy', '?')), missing_b_as_zero=True)


@pytest.mark.parametrize('column, option', [('occupancy', '--zero-occupancy-as-one'),
                                            ('B_iso_or_equiv', '--missing-b-as-zero')])
def test_refusal_names_the_option_that_reads_it(tmp_path, column, option):
    with pytest.raises(io.InputError, match=r'; give it, or use %s$' % option):
        io.read_model(write(tmp_path, with_value(column, '.')))


def test_atom_without_occupancy_or_b_names_both_options(tmp_path):
    # as 9A1N comes: one refusal names both, rather than one after the other
    path = write(tmp_path, with_values({(2, 'occupancy'): '.', (2, 'B_iso_or_equiv'): '?'}))
    with pytest.raises(io.InputError, match=r"atom ATOM 2 C CA .* has '\.' in _atom_site\.occupancy and '\?' in "
                                            r"_atom_site\.B_iso_or_equiv, .*; give them, or use "
                                            r"--zero-occupancy-as-one --missing-b-as-zero$"):
        io.read_model(path)
    with pytest.raises(io.InputError, match=r"has '\?' in _atom_site\.B_iso_or_equiv, .*use --missing-b-as-zero$"):
        io.read_model(path, zero_occupancy_as_one=True)
    assert io.read_model(path, zero_occupancy_as_one=True, missing_b_as_zero=True).n_atoms == 6


def test_no_option_offered_when_one_cannot_help(tmp_path):
    path = write(tmp_path, with_values({(2, 'Cartn_x'): '?', (2, 'B_iso_or_equiv'): '.'}))
    with pytest.raises(io.InputError) as e:
        io.read_model(path)
    assert 'Cartn_x' in str(e.value) and 'B_iso_or_equiv' in str(e.value) and 'use --' not in str(e.value)


def test_unchecked_read_takes_missing_values(tmp_path):
    # for scripts that measure or rewrite models (make_fixtures --fill, corpus_properties)
    m = io.read_model(write(tmp_path, with_value('B_iso_or_equiv', '?')), check=False)
    assert m.n_atoms == 6


@pytest.mark.parametrize('column', ['occupancy', 'B_iso_or_equiv', 'id', 'type_symbol', 'label_alt_id',
                                    'label_entity_id', 'pdbx_PDB_model_num'])
def test_missing_column_is_refused(tmp_path, column):
    i = COLUMNS.index('_atom_site.' + column)
    head = HEAD.replace('_atom_site.%s\n' % column, '')
    # whole rows (one is written over two lines)
    tokens = io.cif_tokens(' '.join(line for line in CLEAN.splitlines() if line != '#'))
    n = len(COLUMNS)
    rows = ''.join(' '.join(io.cif_value(t) for k, t in enumerate(tokens[r:r + n]) if k != i) + '\n'
                   for r in range(0, len(tokens), n)) + '#\n'
    with pytest.raises(io.InputError, match=column):
        io.read_model(write(tmp_path, head + rows))
    assert io.read_model(write(tmp_path, head + rows), check=False).n_atoms == 6
