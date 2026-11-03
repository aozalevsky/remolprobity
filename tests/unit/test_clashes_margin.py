import pytest
from scipy.spatial import cKDTree

from remolprobity.clashes import Group, MarginError, _atoms_by_residue, _cliques, check_cliques, moved_points

CORE = cKDTree([[0.0, 0.0, 0.0]])
POINTS = {
    'near': [(3.0, 0.0, 0.0)], 'far': [(20.0, 0.0, 0.0)], 'owned': [(0.0, 0.0, 0.0)],
    'inside': [(8.0, 0.0, 0.0)], 'far2': [(30.0, 0.0, 0.0)], 'none': [],
}


def run(clique, margin=15.0, core=CORE):
    check_cliques([clique], lambda m: POINTS[m], core, margin, lambda m: 'res ' + m)


def test_clique_touching_core_with_far_member_fails():
    with pytest.raises(MarginError) as e:
        run(['near', 'far'])
    assert 'or more' in str(e.value) and '--margin 25 or more' in str(e.value)
    assert 'res near' in str(e.value)


def test_far_clique_not_touching_core_is_fine():
    run(['far', 'far2'])


def test_owned_clique_inside_margin_is_fine():
    run(['owned', 'inside'])


def test_a_member_without_points_is_an_error():
    # fail closed: a member that matches no atom would leave the clique unchecked
    for clique in (['none'], ['near', 'none'], ['far', 'none']):
        with pytest.raises(RuntimeError, match='res none'):
            run(clique)


def test_no_core_skips_check():
    run(['near', 'far'], core=None)


def test_margin_error_carries_the_margin_it_asks_for():
    # the run retries the block at this margin, without reading the message
    with pytest.raises(MarginError) as e:
        run(['near', 'far'])
    assert e.value.needed == 25
    assert '--margin %d or more' % e.value.needed in str(e.value)



def test_a_block_asks_for_the_largest_margin_any_clique_needs():
    # one retry at the largest need, not one per clique: the first clique
    # needs 25 Å, the second 35 Å
    with pytest.raises(MarginError) as e:
        check_cliques([['near', 'far'], ['near', 'far2']], lambda m: POINTS[m], CORE, 15.0, lambda m: 'res ' + m)
    assert e.value.needed == 35
    assert 'res far2' in str(e.value) and '--margin 35 or more' in str(e.value)

# reduce's USER  MOD records with residue numbers above 9999: rotatable groups
# in hybrid-36 (A0L6 is 10762), flips (amide, His) in decimal of any width
RECORDS = """\
USER  MOD Set 1.1:L1A0L6 TYR OH  :   rot -107:sc=   0.357
USER  MOD Set 1.2:L1A0MA CYS SG  :   rot -143:sc=  0.0679
USER  MOD Set 2.1:L110742 ASN     :      amide:sc=    -2.2! K(o=-10!,f=-7)
USER  MOD Set 2.2:L1A0LQ CYS SG  :   rot  152:sc=    -8.1!
USER  MOD Set 3.1:L1 762 TYR OH  :   rot -107:sc=   0.357
USER  MOD Set 3.2:L1 781AHIS     :     no HD1:sc=   -3.07! K(o=-2.6!,f=-0.89)
USER  MOD Set 3.3:L1  -5 SER OG A:   rot  180:sc=       0
USER  MOD Set 3.4:L1123456BGLN     :      amide:sc=   -3.38! C(o=-7.2!,f=-5.9!)
USER  MOD Set11.1: L10742 ASN     :      amide:sc=   -1.99! K(o=-9.8!,f=-6.8)
USER  MOD Set11.2: LA0LQ CYS SG  :   rot  154:sc=   -7.77!
USER  MOD Single : L10082 ASN     :      amide:sc=   -9.33! C(o=-9.3!,f=-9.2!)
ATOM      1  N   ASN L1074       1.000   2.000   3.000  1.00  0.00           N
"""


def test_cliques_take_both_styles_of_residue_number(tmp_path):
    p = tmp_path / 'withh.pdb'
    p.write_text(RECORDS)
    assert _cliques(str(p)) == [
        [('L1', '10762', 'TYR', 'OH', ''), ('L1', '10802', 'CYS', 'SG', '')],
        [('L1', '10742', 'ASN', '', ''), ('L1', '10782', 'CYS', 'SG', '')],
        [('L1', '762', 'TYR', 'OH', ''), ('L1', '781A', 'HIS', '', ''), ('L1', '-5', 'SER', 'OG', 'A'),
         ('L1', '123456B', 'GLN', '', '')],
        [('L', '10742', 'ASN', '', ''), ('L', '10782', 'CYS', 'SG', '')],
    ]


def test_a_record_without_a_residue_number_is_an_error(tmp_path):
    p = tmp_path / 'withh.pdb'
    p.write_text('USER  MOD Set 1.1:L1 7#2 TYR OH  :   rot -107:sc=   0.357\n')
    with pytest.raises(RuntimeError, match='7#2'):
        _cliques(str(p))


# a serine, an asparagine and a ligand, as _atoms_by_residue keeps a protonated block
SER = [('N', '', (0.0, 0.0, 0.0)), ('CA', '', (1.5, 0.0, 0.0)), ('HA', '', (1.8, -1.0, 0.0)),
       ('CB', '', (2.0, 1.4, 0.0)), ('OG', '', (3.4, 1.5, 0.0)), ('HG', '', (3.7, 2.4, 0.0))]
ASN = [('N', '', (0.0, 5.0, 0.0)), ('CA', '', (1.5, 5.0, 0.0)), ('CB', '', (2.0, 6.4, 0.0)),
       ('CG', '', (3.5, 6.5, 0.0)), ('OD1', '', (4.2, 5.5, 0.0)), ('ND2', '', (4.0, 7.7, 0.0)),
       ('HD21', '', (5.0, 7.8, 0.0)), ('HD22', '', (3.5, 8.6, 0.0))]
ALT = [('CA', '', (0.0, 0.0, 9.0)), ('OG', 'A', (1.0, 0.0, 9.0)), ('HG', 'A', (1.5, 0.8, 9.0)),
       ('OG', 'B', (-1.0, 0.0, 9.0)), ('HG', 'B', (-1.5, 0.8, 9.0))]
# a long ligand: a methyl at one end, its other atoms up to 22 Å away
LIGAND = [('CM', '', (3.0, 0.0, 0.0)), ('HM1', '', (3.5, 0.9, 0.0)), ('HM2', '', (3.5, -0.9, 0.0)),
          ('HM3', '', (2.0, 0.0, 0.0))] + [('C%d' % i, '', (3.0 + 1.5 * i, 0.0, 0.0)) for i in range(1, 14)]
ATOMS = {('A', '5'): SER, ('A', '6'): ASN, ('A', '7'): ALT, ('L', '1'): LIGAND}


def points(atoms):
    return sorted(p for _, _, p in atoms)


def test_a_rotated_group_moves_its_atom_and_its_hydrogens():
    assert sorted(moved_points(ATOMS, Group('A', '5', 'SER', 'OG', ''))) == [(3.4, 1.5, 0.0), (3.7, 2.4, 0.0)]


def test_a_flip_moves_the_side_chain_end():
    assert sorted(moved_points(ATOMS, Group('A', '6', 'ASN', '', ''))) == points(ASN[3:])


def test_a_group_with_an_altloc_moves_that_conformer_only():
    assert sorted(moved_points(ATOMS, Group('A', '7', 'SER', 'OG', 'A'))) == [(1.0, 0.0, 9.0), (1.5, 0.8, 9.0)]


def test_a_group_without_an_altloc_moves_every_conformer():
    assert sorted(moved_points(ATOMS, Group('A', '7', 'SER', 'OG', ''))) == points(ALT[1:])


def test_a_group_not_found_counts_the_whole_residue():
    # fail safe: an atom name or a flip that isn't known counts every atom of the residue
    assert sorted(moved_points(ATOMS, Group('A', '5', 'SER', 'OXT', ''))) == points(SER)
    assert sorted(moved_points(ATOMS, Group('A', '6', 'XYZ', '', ''))) == points(ASN)


def test_a_residue_not_found_has_no_points():
    # check_cliques then refuses the clique, as before
    assert moved_points(ATOMS, Group('B', '1', 'SER', 'OG', '')) == []


def test_a_long_ligand_counts_by_the_methyl_reduce_rotates():
    # with its whole residue the clique would reach 22.5 Å from the core,
    # past the 20 Å a 25 Å margin allows; what reduce moves stays within 4 Å
    clique = [Group('L', '1', 'LIG', 'CM', ''), Group('A', '5', 'SER', 'OG', '')]
    check_cliques([clique], lambda m: moved_points(ATOMS, m), CORE, 25.0, str)
    with pytest.raises(MarginError):
        check_cliques([clique], lambda m: [p for _, _, p in ATOMS[(m.chain, m.resid)]], CORE, 25.0, str)


def test_atoms_by_residue_keeps_names_and_altlocs():
    xyz = {('A', '5', '', 'SER', 'OG', 'A'): (1.0, 2.0, 3.0), ('A', '5', '', 'SER', 'CA', ''): (0.0, 0.0, 0.0),
           ('A', 'A0L6', '', 'TYR', 'OH', ''): (4.0, 5.0, 6.0)}
    assert _atoms_by_residue(xyz) == {('A', '5'): [('OG', 'A', (1.0, 2.0, 3.0)), ('CA', '', (0.0, 0.0, 0.0))],
                                      ('A', '10762'): [('OH', '', (4.0, 5.0, 6.0))]}
