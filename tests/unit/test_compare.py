import json
import pickle

from remolprobity import compare, result


def test_identical():
    a = {'x': [('A', 1, 0.5)], 'n': 3}
    assert compare.differences(a, a) == []


def test_tolerance():
    assert compare.differences({'v': 1.0}, {'v': 1.0 + 1e-12}) == []
    assert compare.differences({'v': 1.0}, {'v': 1.001}, tol=1e-6)


def test_counts_are_exact():
    assert compare.differences({'n': 3}, {'n': 4})


def test_record_lists_report_identities():
    a = {'r': [('A', '1', 0.5), ('A', '2', 0.7)]}
    b = {'r': [('A', '1', 0.5)]}
    d = compare.differences(a, b)
    assert any("('A', '2')" in x and 'first only' in x for x in d)


A1 = ('A', '2', 'SER', 'OG', '')
A2 = ('B', '1', 'ALA', 'CB', '')


def sample(**changes):
    r = {'schema': 'remolprobity.result', 'version': 1,
         'provenance': {'remolprobity': '0.1.0', 'blocks': {'count': 3}, 'input': {'atoms': 40}},
         'ramachandran': {'total': 1, 'favored': 1, 'allowed': 0, 'outliers': 0,
                          'records': [('A', '2', 'SER', '', 'Favored', -60.0, -40.0, 0.9)]},
         'rotamers': {'total': 1, 'favored': 0, 'allowed': 0, 'outliers': 1,
                      'records': [('A', '2', 'SER', '', 'OUTLIER', 'OUTLIER', 0.1)]},
         'clashes': {'clashscore': 50.0, 'atoms': 20, 'records': [(A1, A2, 0.5, 2.1)]},
         'geometry': {'v34': {'bonds': {'total': 2, 'outliers': [(A1, A2, 1.9, 1.5, 0.02, -20.0)]},
                              'angles': {'total': 3, 'outliers': []}},
                      'v33': {'bonds': {'total': 2, 'outliers': [(A1, A2, 1.9, 1.5, 20.0)]},
                              'angles': {'total': 3, 'outliers': []}}},
         'residues': [{'chain': 'A', 'resid': '2', 'resname': 'SER', 'v34': {'bonds': [2, 1.0], 'angles': [3, 2.0]}}]}
    for path, value in changes.items():
        *keys, last = path.split('.')
        d = r
        for k in keys:
            d = d[int(k)] if k.isdigit() else d[k]
        d[int(last) if last.isdigit() else last] = value
    return r


def test_differences_name_the_fields_of_records():
    a = sample()
    b = sample(**{'ramachandran.records.0': ('A', '2', 'SER', '', 'Favored', -60.0, -40.0, 0.8),
                  'clashes.records.0': ((('A', '2', 'SER', 'OG1', '')), A2, 0.5, 2.0),
                  'geometry.v34.bonds.outliers.0': (A1, A2, 1.9, 1.5, 0.02, -21.0),
                  'geometry.v33.bonds.outliers.0': (A1, A2, 1.9, 1.5, 21.0)})
    d = compare.differences(result.comparable(a), result.comparable(b), tol=1e-6)
    assert d == ['.clashes.records[0].atom1.name: %r != %r' % ('OG', 'OG1'),
                 '.clashes.records[0].distance: 2.1 != 2.0',
                 '.geometry.v33.bonds.outliers[0].z: 20.0 != 21.0',
                 '.geometry.v34.bonds.outliers[0].z: -20.0 != -21.0',
                 '.ramachandran.records[0].score: 0.9 != 0.8']


def write_both(tmp_path, a, b):
    pa, pb = tmp_path / 'a.json', tmp_path / 'b.pkl'
    pa.write_text(json.dumps(a))
    pb.write_bytes(pickle.dumps(b, protocol=4))
    return str(pa), str(pb)


def test_cli_same(tmp_path, capsys):
    a, b = write_both(tmp_path, sample(), sample(**{'provenance.blocks.count': 1}))
    assert compare.main([a, b]) == 0
    out = capsys.readouterr().out
    assert 'same' in out and 'clashscore' in out and '50.00' in out


def test_cli_differences_and_limit(tmp_path, capsys):
    b = sample(**{'clashes.clashscore': 51.0, 'rotamers.outliers': 0})
    a, b = write_both(tmp_path, sample(), b)
    assert compare.main([a, b, '--limit', '1']) == 1
    out = capsys.readouterr().out
    assert 'clashes.clashscore: 50.0 != 51.0' in out and 'rotamers.outliers' not in out
    assert 'more' in out


def test_cli_tolerance(tmp_path):
    a, b = write_both(tmp_path, sample(), sample(**{'clashes.clashscore': 50.0 + 1e-8}))
    assert compare.main([a, b]) == 0
    assert compare.main([a, b, '--tol', '1e-12']) == 1


def test_cli_json(tmp_path, capsys):
    a, b = write_both(tmp_path, sample(), sample(**{'clashes.clashscore': 51.0}))
    assert compare.main([a, b, '--json']) == 1
    out = json.loads(capsys.readouterr().out)
    assert out['same'] is False and out['differences'] == ['clashes.clashscore: 50.0 != 51.0']
    assert out['summary'][0]['clashscore'] == 50.0 and out['summary'][1]['clashscore'] == 51.0


def test_cli_text_names_no_geometry_definition(tmp_path, capsys):
    a, b = write_both(tmp_path, sample(), sample())
    assert compare.main([a, b]) == 0
    out = capsys.readouterr().out
    assert 'v33' not in out and 'v34' not in out
    assert not [k for k in compare.summary(sample()) if 'v3' in k]


def test_summary_has_one_clashscore_per_result():
    stock = sample()
    stock['provenance'] = {'stock': True, 'clashscore': 49.0}
    s = compare.summary(stock)
    assert s['clashscore'] == 50.0 and 49.0 not in s.values()


def test_cli_geometry_in_one_only_is_not_compared(tmp_path, capsys):
    b = sample()
    del b['geometry']['v33']
    a, b = write_both(tmp_path, sample(), b)
    assert compare.main([a, b]) == 0
    assert 'v33' not in capsys.readouterr().out
    assert compare.main([a, b, '--json']) == 0
    assert json.loads(capsys.readouterr().out)['not_compared'] == ['geometry v33 is only in the first: not compared']
    c = sample()
    del c['geometry']['v34']
    del c['residues'][0]['v34']
    a, c = write_both(tmp_path, sample(), c)
    assert compare.main([a, c]) == 0


def test_cli_unreadable(tmp_path, capsys):
    p = tmp_path / 'x.json'
    p.write_text('{}')
    assert compare.main([str(p), str(p)]) == 2
    assert 'not a remolprobity result' in capsys.readouterr().err
