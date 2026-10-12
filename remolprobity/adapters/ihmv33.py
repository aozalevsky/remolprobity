"""
IHMValidation 3.3's MolProbity cache (<stem>.mp.pkl), as the code in the
2026 images reads it: a bare {model id: data}, bonds and angles from
molprobity.molprobity's restraints validation (the v33 definition).

Two differences from 3.3's own converter, both in names only: residue
numbers are decimal (3.3 writes cctbx's resid, hybrid-36 above 9999: A000
for 10000), and clash atoms carry the model's chain and residue names (3.3
writes the PDB names probe saw when the model doesn't fit PDB format).
"""
import argparse
import os
import pickle

from .. import result as R


def _a(atom):
    return atom[:4]


def model_data(result):
    if 'v33' not in result['geometry']:
        raise ValueError('the result has no v33 geometry; run with --geometry v33')
    rama, rota, v33 = result['ramachandran'], result['rotamers'], result['geometry']['v33']

    def geometry(kind, n):
        lst = [tuple(_a(x) for x in o[:n]) + o[n:] for o in v33[kind]['outliers']]
        return {'total': v33[kind]['total'], 'outliers': len(lst), 'outliers_list': lst}

    counts = ('total', 'favored', 'allowed', 'outliers')
    return {
        'clash': {'clashscore': result['clashes']['clashscore'],
                  'clashes_list': [(_a(a), _a(b), o) for a, b, o, _ in result['clashes']['records']]},
        'bonds': geometry('bonds', 2),
        'angles': geometry('angles', 3),
        'rama': dict({k: rama[k] for k in counts},
                     outliers_list=[(c, i, n, s) for c, i, n, _, cls, _, _, s in rama['records'] if cls == 'OUTLIER']),
        'rota': dict({k: rota[k] for k in counts},
                     outliers_list=[(c, i, n, s) for c, i, n, _, ev, _, s in rota['records'] if ev == 'OUTLIER']),
    }


def cache(result, model_id):
    return {int(model_id): model_data(result)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('model_id', type=int)
    p.add_argument('out')
    a = p.parse_args(argv)
    data = cache(R.read(a.result), a.model_id)
    temp = '%s.%d.tmp' % (a.out, os.getpid())
    with open(temp, 'wb') as f:
        pickle.dump(data, f)
    os.replace(temp, a.out)


if __name__ == '__main__':
    main()
