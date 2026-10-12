#!/usr/bin/env python3
"""
Make the corpus in DIR: download PDB entries, derive variants, check SHA-256
(recording it the first time) and properties. Files already in DIR are
kept. REMOLPROBITY_LOCAL_<NAME>=path copies a local file instead of
downloading. usage: fetch_corpus.py [--dir DIR] [--tier oracle|heavy|perf]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_fixtures  # noqa: E402
from corpus_properties import properties  # noqa: E402

CORPUS = os.path.join(REPO, 'tests', 'corpus', 'corpus.json')
TIERS = ('oracle', 'heavy', 'perf')
URL = 'https://files.rcsb.org/download/%s.cif'
AS_PDB = ('import iotbx.pdb, sys; h = iotbx.pdb.input(file_name=sys.argv[1]).construct_hierarchy(); '
          'open(sys.argv[2], "w").write(h.as_pdb_string())')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def satisfied(prop, p):
    if '>' in prop:
        k, v = prop.split('>')
        return p[k] > float(v)
    if '=' in prop:
        k, v = prop.split('=')
        return str(p[k]) == v
    return bool(p[prop])


def derive(recipe, src, out):
    kind, args = recipe[0], recipe[2:]
    if kind == 'renumber':
        make_fixtures.renumber(src, out, args[0], args[1])
    elif kind == 'zero-occupancy':
        make_fixtures.zero_occupancy(src, out, args[0])
    elif kind == 'subset':
        make_fixtures.subset(src, out, args[0], fill=True)
    elif kind == 'tile':
        make_fixtures.tile(src, out, args[0], fill=True)
    elif kind == 'as-pdb':
        subprocess.run(['apptainer', 'exec', os.environ['REMOLPROBITY_IMAGE'], 'mmtbx.python', '-c', AS_PDB,
                        src, out], check=True)
    else:
        raise ValueError('unknown recipe %r' % kind)


def fetch(entry, d, paths):
    ext = '.pdb' if entry.get('recipe', [''])[0] == 'as-pdb' else '.cif'
    out = os.path.join(d, entry['name'] + ext)
    if os.path.exists(out):
        return out
    temp = os.path.join(d, '.part.' + entry['name'] + ext)
    local = os.environ.get('REMOLPROBITY_LOCAL_%s' % entry['name'].upper())
    if entry['source'] == 'rcsb':
        if local:
            shutil.copyfile(local, temp)
        else:
            urllib.request.urlretrieve(URL % entry['id'], temp)
    else:
        derive(entry['recipe'], paths[entry['recipe'][1]], temp)
    os.replace(temp, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', default=os.environ.get('REMOLPROBITY_CORPUS',
                                                   os.path.expanduser('~/.cache/remolprobity/corpus')))
    p.add_argument('--tier', choices=TIERS, default='oracle')
    a = p.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    with open(CORPUS) as f:
        entries = json.load(f)
    paths, ok = {}, True
    for e in entries:
        if TIERS.index(e.get('tier', 'oracle')) > TIERS.index(a.tier):
            continue
        path = paths[e['name']] = fetch(e, a.dir, paths)
        digest = sha(path)
        if not e['sha256']:
            e['sha256'] = digest
        elif e['sha256'] != digest:
            print('%s: SHA-256 changed (was %s)' % (e['name'], e['sha256'][:12]))
            ok = False
        if e.get('tier', 'oracle') == 'oracle':
            props = properties(path)
            missing = [x for x in e['properties'] if not satisfied(x, props)]
            print('%-16s %9d atoms  %s' % (e['name'], props['atoms'], 'MISSING %s' % missing if missing else 'ok'))
            ok &= not missing
        else:
            print('%-16s %9.0f MB' % (e['name'], os.path.getsize(path) / 1e6))
    with open(CORPUS, 'w') as f:
        json.dump(entries, f, indent=1)
        f.write('\n')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
