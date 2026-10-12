import io
import json
import os
import pickle
import sys

import pytest

from remolprobity import result

RES = [('A', '1'), ('A', '2'), ('B', '1')]
NAMES = ['GLY', 'SER', 'ALA']
A1 = ('A', '2', 'SER', 'OG', '')
A2 = ('B', '1', 'ALA', 'CB', '')


def block(b, owned, rama=(), rota=(), clashes=(), probe_atoms=0, outliers=()):
    geo = {'v34': {'bonds': {'total': 2 * len(owned), 'outliers': list(outliers)},
                   'angles': {'total': 3 * len(owned), 'outliers': []},
                   'residues': {k: [2, 1.0, 3, 2.0] for k in owned}}}
    return {'block': b, 'owned': list(owned), 'ramachandran': list(rama), 'rotamers': list(rota),
            'clashes': list(clashes), 'probe_atoms': probe_atoms, 'geometry': geo}


class M:
    residues = RES
    resnames = NAMES


def blocks():
    return [
        block(0, [('A', '1'), ('A', '2')],
              rama=[('A', '2', 'SER', 'B', 'OUTLIER', 60.0, 60.0, 0.01),
                    ('A', '2', 'SER', 'A', 'Favored', -60.0, -40.0, 0.9)],
              rota=[('A', '2', 'SER', '', 'OUTLIER', 'OUTLIER', 0.1)],
              clashes=[(A1, A2, 0.5, 2.1)], probe_atoms=20,
              outliers=[(A1, A2, 1.9, 1.5, 0.02, -20.0)]),
        block(1, [('B', '1')], rota=[('B', '1', 'ALA', '', 'Favored', 'm', 50.0)], probe_atoms=10),
    ]


def test_counting_rules():
    rama = [('A', '1', 'X', '', 'Favored', 0, 0, 1), ('A', '2', 'X', 'A', 'Allowed', 0, 0, 1),
            ('A', '2', 'X', 'B', 'OUTLIER', 0, 0, 1)]
    assert result.count_ramachandran(rama) == {'total': 2, 'favored': 1, 'allowed': 1, 'outliers': 0}
    rota = [('A', '2', 'X', 'A', 'OUTLIER', 'OUTLIER', 0), ('A', '2', 'X', 'B', 'Favored', 'm', 1)]
    assert result.count_rotamers(rota) == {'total': 2, 'favored': 1, 'allowed': 0, 'outliers': 1}


def test_merge_is_canonical_and_complete():
    r = result.merge(M, blocks(), ['v34'], {'x': 1})
    r2 = result.merge(M, list(reversed(blocks())), ['v34'], {'x': 1})
    assert pickle.dumps(r) == pickle.dumps(r2)
    assert r['schema'] == result.SCHEMA and r['version'] == result.VERSION
    assert [x[3] for x in r['ramachandran']['records']] == ['A', 'B']
    assert r['ramachandran']['total'] == 1 and r['ramachandran']['favored'] == 1
    assert r['clashes'] == {'clashscore': 1000 / 30, 'atoms': 30, 'records': [(A1, A2, 0.5, 2.1)]}
    assert r['geometry']['v34']['bonds']['total'] == 6
    assert [e['chain'] + e['resid'] for e in r['residues']] == ['A1', 'A2', 'B1']
    assert r['residues'][1]['v34'] == {'bonds': [2, 1.0], 'angles': [3, 2.0]}


def test_merge_refuses_double_ownership():
    b = blocks()
    b[1]['owned'].append(('A', '1'))
    with pytest.raises(result.MergeError, match='owned by blocks'):
        result.merge(M, b, ['v34'], {})


def test_merge_refuses_unowned_residues():
    b = blocks()
    b[1]['owned'] = []
    with pytest.raises(result.MergeError, match='owned by no block'):
        result.merge(M, b, ['v34'], {})


def test_write_read_and_comparable(tmp_path):
    r = result.merge(M, blocks(), ['v34'], {'wall': 1.0})
    p = str(tmp_path / 'r.pkl')
    result.write(r, p)
    assert result.read(p) == r
    assert 'provenance' not in result.comparable(r)


def test_read_refuses_other_files(tmp_path):
    p = tmp_path / 'x.pkl'
    p.write_bytes(pickle.dumps({'schema': 'other'}))
    with pytest.raises(ValueError, match='not a remolprobity result'):
        result.read(str(p))


def test_no_probe_atoms_gives_zero_clashscore():
    b = blocks()
    for x in b:
        x['probe_atoms'] = 0
        x['clashes'] = []
    assert result.merge(M, b, ['v34'], {})['clashes']['clashscore'] == 0.0


def test_block_cannot_report_rotamer_for_another_blocks_residue():
    b = blocks()
    b[1]['rotamers'].append(('A', '2', 'SER', '', 'Favored', 'm', 1.0))
    with pytest.raises(result.MergeError, match='block 1'):
        result.merge(M, b, ['v34'], {})


def test_block_cannot_report_clash_for_another_blocks_residue():
    b = blocks()
    b[1]['clashes'].append((A1, A2, 0.3, 2.5))  # first atom A1 is in ('A', '2'), owned by block 0
    with pytest.raises(result.MergeError, match='block 1'):
        result.merge(M, b, ['v34'], {})


def test_block_cannot_report_outlier_for_another_blocks_residue():
    b = blocks()
    b[1]['geometry']['v34']['bonds']['outliers'].append((A1, A2, 1.9, 1.5, 0.02, -20.0))
    with pytest.raises(result.MergeError, match='block 1'):
        result.merge(M, b, ['v34'], {})


def test_block_cannot_report_residue_geometry_for_another_blocks_residue():
    b = blocks()
    b[1]['geometry']['v34']['residues'][('A', '2')] = [1, 0.5, 1, 0.5]
    with pytest.raises(result.MergeError, match='block 1'):
        result.merge(M, b, ['v34'], {})


def test_residue_geometry_for_a_residue_not_in_the_model_is_refused():
    b = blocks()
    b[1]['geometry']['v34']['residues'][('C', '9')] = [1, 0.5, 1, 0.5]
    with pytest.raises(result.MergeError, match='no block'):
        result.merge(M, b, ['v34'], {})


def test_ties_keep_a_total_order():
    def tied(clashes, outliers):
        b = blocks()
        b[0]['clashes'] = clashes
        b[0]['geometry']['v34']['bonds']['outliers'] = outliers
        return b

    c1 = (A1, A2, 0.5, 2.1)
    c2 = (A1, A2, 0.7, 2.0)
    o1 = (A1, A2, 1.9, 1.5, 0.02, -20.0)
    o2 = (A1, A2, 2.1, 1.5, 0.02, 5.0)
    r1 = result.merge(M, tied([c1, c2], [o1, o2]), ['v34'], {})
    r2 = result.merge(M, tied([c2, c1], [o2, o1]), ['v34'], {})
    assert pickle.dumps(r1) == pickle.dumps(r2)


def test_merge_refuses_a_block_asking_for_a_larger_margin():
    b = blocks()
    b[1] = {'block': 1, 'margin_needed': 23, 'message': 'a reduce clique ...'}
    with pytest.raises(result.MergeError, match='block 1 .*margin'):
        result.merge(M, b, ['v34'], {})


def full_result():
    r = result.merge(M, blocks(), ['v34'], {'blocks': {'count': 2, 'retried': {1: 30}}, 'wall': 1.0})
    r['geometry']['v33'] = {'bonds': {'total': 4, 'outliers': [(A1, A2, 1.9, 1.5, 20.0)]},
                            'angles': {'total': 0, 'outliers': []}}
    return r


def test_json_round_trip_equals_the_pickle(tmp_path):
    r = full_result()
    result.write(r, str(tmp_path / 'r.json'))
    result.write(r, str(tmp_path / 'r.pkl'))
    assert json.load(open(tmp_path / 'r.json'))['schema'] == result.SCHEMA
    assert result.read(str(tmp_path / 'r.json')) == r == result.read(str(tmp_path / 'r.pkl'))


def test_format_comes_from_the_content_not_the_name(tmp_path):
    r = full_result()
    p = tmp_path / 'r.json'
    result.write(r, str(p))
    os.replace(p, tmp_path / 'r.out')
    assert result.read(str(tmp_path / 'r.out')) == r


def test_dash_writes_json_to_the_stream_and_reads_stdin(monkeypatch):
    r = full_result()
    out = io.StringIO()
    result.write(r, '-', out=out)
    assert json.loads(out.getvalue())['clashes']['atoms'] == 30
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(out.getvalue().encode())))
    assert result.read('-') == r


def test_read_refuses_json_that_is_not_a_result(tmp_path):
    p = tmp_path / 'x.json'
    p.write_text('{"schema": "other"}')
    with pytest.raises(ValueError, match='not a remolprobity result'):
        result.read(str(p))
    p.write_text('not json')
    with pytest.raises(ValueError, match='not a remolprobity result'):
        result.read(str(p))
