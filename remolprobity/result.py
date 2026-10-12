"""The result of a run: schema, counting rules, merging blocks, reading and writing"""
import json
import os
import pickle
import sys

SCHEMA = 'remolprobity.result'
VERSION = 1
DEVIATIONS = (
    'reduce has the clique-order patch: its result for a group does not depend '
    'on unrelated parts of the model',
    'restraints are built without the NCS shortcut',
    'clash atoms are named as in the model, not as the PDB names probe saw',
)
_CLASS = {'Favored': 'favored', 'Allowed': 'allowed', 'OUTLIER': 'outliers'}


class MergeError(Exception):
    """Block results that don't add up to the model"""


def count_ramachandran(records):
    """As ramalyze counts: each residue once, its first record with altloc '' or 'A'"""
    counts = {'total': 0, 'favored': 0, 'allowed': 0, 'outliers': 0}
    seen = set()
    for chain, resid, _, altloc, cls, _, _, _ in records:
        if altloc in ('', 'A') and (chain, resid) not in seen:
            seen.add((chain, resid))
            counts['total'] += 1
            counts[_CLASS[cls]] += 1
    return counts


def count_rotamers(records):
    """As rotalyze counts: every record"""
    counts = {'total': len(records), 'favored': 0, 'allowed': 0, 'outliers': 0}
    for r in records:
        counts[_CLASS[r[4]]] += 1
    return counts


def _atoms(outlier):
    return outlier[:3] if isinstance(outlier[2], tuple) else outlier[:2]


def assemble(residues, resnames, ramachandran, rotamers, clashes, probe_atoms, geometry,
             residue_geometry, provenance, ramachandran_counts=None, rotamer_counts=None):
    """
    A result from its parts, in canonical order: records by residue in file
    order then altloc, clashes by atom pair, outliers by atoms. residues:
    (chain, resid) in file order; geometry: {definition: {'bonds'/'angles':
    {'total', 'outliers'}}}; residue_geometry: {(chain, resid): [nb, z2b,
    na, z2a]} for v34. Counts are computed unless given.
    """
    index = {k: i for i, k in enumerate(residues)}
    rama = sorted(ramachandran, key=lambda r: (index[(r[0], r[1])], r[3], r))
    rota = sorted(rotamers, key=lambda r: (index[(r[0], r[1])], r[3], r))
    clash = sorted(clashes, key=lambda c: (c[0], c[1], c[2], c[3]))
    geo = {d: {k: {'total': g[k]['total'],
                   'outliers': sorted(g[k]['outliers'], key=lambda o: (_atoms(o), o[len(_atoms(o)):]))}
               for k in ('bonds', 'angles')} for d, g in geometry.items()}
    entries = []
    for key, name in zip(residues, resnames):
        entry = {'chain': key[0], 'resid': key[1], 'resname': name}
        if 'v34' in geometry:
            nb, zb, na, za = residue_geometry.get(key, (0, 0.0, 0, 0.0))
            entry['v34'] = {'bonds': [nb, zb], 'angles': [na, za]}
        entries.append(entry)
    return {
        'schema': SCHEMA, 'version': VERSION, 'provenance': provenance,
        'ramachandran': dict(ramachandran_counts or count_ramachandran(rama), records=rama),
        'rotamers': dict(rotamer_counts or count_rotamers(rota), records=rota),
        'clashes': {'clashscore': len(clash) * 1000 / probe_atoms if probe_atoms else 0.0,
                    'atoms': probe_atoms, 'records': clash},
        'geometry': geo,
        'residues': entries,
    }


def _check_owned(owners, br, key, what):
    """Refuse a record of block br about residue key unless br owns that residue"""
    owner = owners.get(key)
    if owner != br['block']:
        raise MergeError('block %d reports a %s for residue %s %s, which %s'
                         % (br['block'], what, key[0], key[1],
                            'is owned by no block' if owner is None else 'is owned by block %d' % owner))


def merge(model, block_results, definitions, provenance):
    """Block results into a result, checking every residue is owned exactly once"""
    for br in block_results:
        if 'margin_needed' in br:
            raise MergeError('block %d has no result: it asked for a margin of %s Å'
                             % (br['block'], br['margin_needed']))
    owners = {}
    for br in block_results:
        for key in br['owned']:
            if key in owners:
                raise MergeError('residue %s %s is owned by blocks %d and %d'
                                 % (key + (owners[key], br['block'])))
            owners[key] = br['block']
    missing = [k for k in model.residues if k not in owners]
    if missing:
        raise MergeError('%d residues are owned by no block, e.g. %s %s' % ((len(missing),) + missing[0]))
    if len(owners) != len(model.residues):
        raise MergeError('blocks own residues the model does not have')
    for br in block_results:
        for r in br['ramachandran'] + br['rotamers']:
            _check_owned(owners, br, (r[0], r[1]), 'record')
        for c in br['clashes']:
            _check_owned(owners, br, (c[0][0], c[0][1]), 'clash')
        for g in br['geometry'].values():
            for k in ('bonds', 'angles'):
                for o in g[k]['outliers']:
                    _check_owned(owners, br, (o[0][0], o[0][1]), 'geometry outlier')
        for key in br['geometry'].get('v34', {}).get('residues', {}):
            _check_owned(owners, br, key, 'residue geometry')
    geometry = {d: {k: {'total': sum(br['geometry'][d][k]['total'] for br in block_results),
                        'outliers': [o for br in block_results for o in br['geometry'][d][k]['outliers']]}
                    for k in ('bonds', 'angles')} for d in definitions}
    residue_geometry = {}
    for br in block_results:
        for key, values in br['geometry'].get('v34', {}).get('residues', {}).items():
            residue_geometry[key] = values
    return assemble(model.residues, model.resnames,
                    [r for br in block_results for r in br['ramachandran']],
                    [r for br in block_results for r in br['rotamers']],
                    [c for br in block_results for c in br['clashes']],
                    sum(br['probe_atoms'] for br in block_results),
                    geometry, residue_geometry, provenance)


def write(result, path, out=None):
    """
    Write result at path: JSON for '-' (to out, else stdout) or a name ending
    in .json, else a pickle. A file is replaced only once complete.
    """
    if path == '-':
        out = out or sys.stdout
        json.dump(result, out)
        out.write('\n')
        out.flush()
        return
    temp = '%s.%d.tmp' % (path, os.getpid())
    if path.endswith('.json'):
        with open(temp, 'w') as f:
            json.dump(result, f)
    else:
        with open(temp, 'wb') as f:
            pickle.dump(result, f, protocol=4)
    os.replace(temp, path)


def _tuples(x):
    return tuple(_tuples(y) for y in x) if isinstance(x, list) else x


def _from_json(r):
    """What JSON turned into lists and string keys, as the pickle has them"""
    for kind in ('ramachandran', 'rotamers', 'clashes'):
        if isinstance(r.get(kind), dict) and 'records' in r[kind]:
            r[kind]['records'] = [_tuples(x) for x in r[kind]['records']]
    for definition in r.get('geometry', {}).values():
        for kind in definition.values():
            kind['outliers'] = [_tuples(x) for x in kind['outliers']]
    retried = r.get('provenance', {}).get('blocks', {}).get('retried')
    if retried:
        r['provenance']['blocks']['retried'] = {int(b): m for b, m in retried.items()}
    return r


def read(path):
    """A result from a pickle or JSON file (by its content), or JSON on stdin for '-'"""
    data = sys.stdin.buffer.read() if path == '-' else open(path, 'rb').read()
    try:
        r = pickle.loads(data) if data[:1] == b'\x80' else _from_json(json.loads(data))
    except (pickle.UnpicklingError, ValueError, TypeError, AttributeError, KeyError, EOFError):
        r = None
    if not isinstance(r, dict) or r.get('schema') != SCHEMA:
        raise ValueError('%s is not a remolprobity result' % path)
    if r.get('version') != VERSION:
        raise ValueError('%s is a version %s result; this is version %d' % (path, r.get('version'), VERSION))
    return r


def comparable(result):
    """The result without its provenance (times, paths, versions)"""
    return {k: v for k, v in result.items() if k != 'provenance'}
