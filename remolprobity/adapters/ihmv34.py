"""
IHMValidation 3.4's MolProbity cache (<stem>.mp.pkl): {model id: data} with
provenance, as its cache.wrap_cache writes it; data as its
molprobity_convert.py writes it
"""
import argparse
import os
import pickle
from datetime import datetime, timezone

from .. import result as R

CACHE_FORMAT = 1  # cache.py's CACHE_FORMAT


def _a(atom):
    return atom[:4]


def model_data(result):
    if 'v34' not in result['geometry']:
        raise ValueError('the result has no v34 geometry; run with --geometry v34')
    rama, rota = result['ramachandran'], result['rotamers']
    residues = {}
    for e in result['residues']:
        residues[(e['chain'], e['resid'])] = {
            'chain': e['chain'], 'resid': e['resid'], 'resname': e['resname'], 'rama': None,
            'phi': None, 'psi': None, 'rotamer': None,
            'bonds': list(e['v34']['bonds']), 'angles': list(e['v34']['angles'])}
    for chain, resid, _, _, cls, phi, psi, _ in rama['records']:
        r = residues[(chain, resid)]
        if r['rama'] is None:
            r['rama'], r['phi'], r['psi'] = cls, phi, psi
    for chain, resid, _, _, _, name, _ in rota['records']:
        r = residues[(chain, resid)]
        if r['rotamer'] is None:
            r['rotamer'] = name
    keep = [r for r in residues.values()
            if r['rama'] is not None or r['rotamer'] is not None or r['bonds'][0] or r['angles'][0]]
    v34 = result['geometry']['v34']
    return {
        'clash': {'clashscore': result['clashes']['clashscore'],
                  'clashes_list': [(_a(a), _a(b), o, d) for a, b, o, d in result['clashes']['records']]},
        'rama': dict({k: rama[k] for k in ('total', 'favored', 'allowed', 'outliers')},
                     outliers_list=[(c, i, n, s) for c, i, n, _, cls, _, _, s in rama['records'] if cls == 'OUTLIER']),
        'rota': dict({k: rota[k] for k in ('total', 'favored', 'allowed', 'outliers')},
                     outliers_list=[(c, i, n, s) for c, i, n, _, ev, _, s in rota['records'] if ev == 'OUTLIER']),
        'residues': keep,
        'mp_bonds': {'total': v34['bonds']['total'],
                     'outliers_list': [tuple(_a(x) for x in o[:2]) + o[2:] for o in v34['bonds']['outliers']]},
        'mp_angles': {'total': v34['angles']['total'],
                      'outliers_list': [tuple(_a(x) for x in o[:3]) + o[3:] for o in v34['angles']['outliers']]},
    }


def cache(result, model_id, software=None, ihmv_version=None):
    """
    The cache entry. software: the result's remolprobity, cctbx, reduce and
    probe versions, updated with the given ones; ihmv_version only if given
    """
    prov = result['provenance']
    soft = {k: str(prov[k]) for k in ('remolprobity', 'cctbx', 'reduce', 'probe') if k in prov}
    soft.update(software or {})
    out = {'cache_format': CACHE_FORMAT,
           'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'software': soft, 'data': {int(model_id): model_data(result)}}
    if ihmv_version:
        out['ihmv_version'] = ihmv_version
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('model_id', type=int)
    p.add_argument('out')
    p.add_argument('--software', action='append', default=[], help='KEY=VALUE, repeated')
    p.add_argument('--ihmv-version')
    a = p.parse_args(argv)
    software = dict(kv.split('=', 1) for kv in a.software)
    data = cache(R.read(a.result), a.model_id, software, a.ihmv_version)
    temp = '%s.%d.tmp' % (a.out, os.getpid())
    with open(temp, 'wb') as f:
        pickle.dump(data, f)
    os.replace(temp, a.out)


if __name__ == '__main__':
    main()
