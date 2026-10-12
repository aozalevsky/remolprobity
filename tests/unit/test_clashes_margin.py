import pytest
from scipy.spatial import cKDTree

from remolprobity.clashes import MarginError, _cliques, check_cliques

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
        [('L1', '10762'), ('L1', '10802')],
        [('L1', '10742'), ('L1', '10782')],
        [('L1', '762'), ('L1', '781A'), ('L1', '-5'), ('L1', '123456B')],
        [('L', '10742'), ('L', '10782')],
    ]


def test_a_record_without_a_residue_number_is_an_error(tmp_path):
    p = tmp_path / 'withh.pdb'
    p.write_text('USER  MOD Set 1.1:L1 7#2 TYR OH  :   rot -107:sc=   0.357\n')
    with pytest.raises(RuntimeError, match='7#2'):
        _cliques(str(p))
