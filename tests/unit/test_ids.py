from types import SimpleNamespace

from remolprobity import ids


def test_residue_number_decimal_and_hybrid36():
    assert ids.residue_number(' 42') == 42
    assert ids.residue_number('-3') == -3
    assert ids.residue_number('A000') == 10000
    assert ids.residue_number('ZZZZ') == 1223055
    assert ids.residue_number('a000') == 1223056


def test_residue_id():
    assert ids.residue_id('  42', ' ') == '42'
    assert ids.residue_id('42', 'A') == '42A'
    assert ids.residue_id('A000') == '10000'


def test_keys_of_cctbx_like_objects():
    a = SimpleNamespace(chain_id=' A', resseq='  12', icode='B', resname='ALA', name=' CA ', altloc=' ')
    assert ids.residue_key(a) == ('A', '12B')
    assert ids.atom_tuple(a) == ('A', '12B', 'ALA', 'CA', '')
