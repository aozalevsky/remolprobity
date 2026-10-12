import pickle

import pytest

from remolprobity import result
from remolprobity.adapters import ihmv33, ihmv34

A = ('A', '2', 'SER', 'OG', '')
B = ('B', '1', 'ALA', 'CB', 'A')


def sample(definitions=('v34', 'v33')):
    geometry = {}
    if 'v34' in definitions:
        geometry['v34'] = {'bonds': {'total': 6, 'outliers': [(A, B, 1.9, 1.5, 0.02, -20.0)]},
                           'angles': {'total': 9, 'outliers': []}}
    if 'v33' in definitions:
        geometry['v33'] = {'bonds': {'total': 7, 'outliers': [(A, B, 1.9, 1.5, 20.0)]},
                           'angles': {'total': 9, 'outliers': [(A, B, A, 100.0, 120.0, 4.5)]}}
    return result.assemble(
        [('A', '1'), ('A', '2'), ('B', '1')], ['GLY', 'SER', 'ALA'],
        [('A', '2', 'SER', '', 'OUTLIER', 60.0, 60.0, 0.01)],
        [('B', '1', 'ALA', 'A', 'OUTLIER', 'OUTLIER', 0.1), ('B', '1', 'ALA', 'B', 'Favored', 'm', 50.0)],
        [(A, B, 0.5, 2.1)], 30, geometry,
        {('A', '2'): [2, 400.0, 3, 1.0]}, {'remolprobity': '0.1.0', 'reduce': 'r', 'probe': 'p'})


def test_ihmv34_model_data():
    d = ihmv34.model_data(sample())
    assert d['clash'] == {'clashscore': 1000 / 30, 'clashes_list': [(A[:4], B[:4], 0.5, 2.1)]}
    assert d['rama']['outliers_list'] == [('A', '2', 'SER', 0.01)]
    assert d['rota']['total'] == 2 and d['rota']['outliers_list'] == [('B', '1', 'ALA', 0.1)]
    assert d['mp_bonds'] == {'total': 6, 'outliers_list': [(A[:4], B[:4], 1.9, 1.5, 0.02, -20.0)]}
    res = {(r['chain'], r['resid']): r for r in d['residues']}
    assert set(res) == {('A', '2'), ('B', '1')}  # residues without data are left out
    assert res[('A', '2')]['rama'] == 'OUTLIER' and res[('A', '2')]['bonds'] == [2, 400.0]
    assert res[('B', '1')]['rotamer'] == 'OUTLIER'


def test_ihmv34_cache_wrapper():
    c = ihmv34.cache(sample(), 1, {'MolProbity': '4.5.2'})
    assert c['cache_format'] == 1 and list(c['data']) == [1]
    assert c['software'] == {'remolprobity': '0.1.0', 'reduce': 'r', 'probe': 'p', 'MolProbity': '4.5.2'}
    assert 'ihmv_version' not in c
    assert ihmv34.cache(sample(), 1, ihmv_version='3.4')['ihmv_version'] == '3.4'


def test_ihmv34_needs_v34():
    with pytest.raises(ValueError, match='v34'):
        ihmv34.model_data(sample(('v33',)))


def test_ihmv33_model_data():
    d = ihmv33.cache(sample(), 1)[1]
    assert d['clash']['clashes_list'] == [(A[:4], B[:4], 0.5)]
    assert d['bonds'] == {'total': 7, 'outliers': 1, 'outliers_list': [(A[:4], B[:4], 1.9, 1.5, 20.0)]}
    assert d['angles']['outliers'] == 1
    assert d['rama']['outliers'] == 1


def test_ihmv33_needs_v33():
    with pytest.raises(ValueError, match='v33'):
        ihmv33.model_data(sample(('v34',)))


def test_cli(tmp_path):
    r = str(tmp_path / 'r.pkl')
    result.write(sample(), r)
    out = str(tmp_path / 'c.pkl')
    ihmv33.main([r, '1', out])
    assert list(pickle.load(open(out, 'rb'))) == [1]
    ihmv34.main([r, '1', out, '--software', 'MolProbity=4.5.2', '--ihmv-version', '3.4'])
    c = pickle.load(open(out, 'rb'))
    assert c['software']['MolProbity'] == '4.5.2' and c['ihmv_version'] == '3.4'


def test_cli_reads_json_results(tmp_path):
    for name in ('r.pkl', 'r.json'):
        result.write(sample(), str(tmp_path / name))
        ihmv33.main([str(tmp_path / name), '1', str(tmp_path / (name + '.mp.pkl'))])
    assert pickle.load(open(tmp_path / 'r.pkl.mp.pkl', 'rb')) == pickle.load(open(tmp_path / 'r.json.mp.pkl', 'rb'))
