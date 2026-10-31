"""The result of a run: schema, counting rules, merging blocks, reading and writing"""
import json
import math
import os
import pickle
import re
import sys

SCHEMA = 'remolprobity.result'
VERSION = 2
# restraint kinds by geometry definition
KINDS = {'v34': ('bonds', 'angles'), 'v33': ('bonds', 'angles', 'dihedrals', 'chiralities', 'planarities')}
CBETA_OUTLIER = 0.25  # Å, as cbetadev
OMEGA_COUNTS = ('proline', 'general', 'cis_proline', 'cis_general', 'twisted_proline', 'twisted_general')
# CaBLAM's cutoffs (mmtbx.validation.cablam): outlier and disfavoured CaBLAM scores, CA geometry outliers
CABLAM_OUTLIER, CABLAM_DISFAVORED, CA_GEOM_OUTLIER = 0.01, 0.05, 0.005
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


def molprobity_score(clashscore, rotamers, ramachandran):
    """
    As mmtbx.validation.utils.molprobity_score, from the clashscore and the
    rotamer and Ramachandran counts: their percentages are 0 without
    residues, as stock takes them
    """
    rota_out = 100.0 * rotamers['outliers'] / rotamers['total'] if rotamers['total'] else 0.0
    rama_fav = 100.0 * ramachandran['favored'] / ramachandran['total'] if ramachandran['total'] else 0.0
    return (0.426 * math.log(1 + clashscore) + 0.33 * math.log(1 + max(0, rota_out - 1)) +
            0.25 * math.log(1 + max(0, 100.0 - rama_fav - 2))) + 0.5


def count_cbeta(records):
    """As cbetadev counts: every record; outliers at CBETA_OUTLIER Å or more"""
    return {'total': len(records), 'outliers': sum(1 for r in records if r[4] >= CBETA_OUTLIER)}


def cbeta_outliers(records):
    return [r for r in records if r[4] >= CBETA_OUTLIER]


def filter_omega(model, raw):
    """
    The peptides omegalyze keeps of these (residues.omegas records from the
    blocks), as (chain, resid, resname, altloc, previous resid, previous
    altloc, omega, type). omegalyze keeps a peptide without altlocs only
    from the conformer whose altloc is that of the first peptide it met
    since the chain id last changed. Over the whole model that is the first
    conformer (model.conformer0) of the first chain object, among those
    with peptides, of each stretch of chain objects with one chain id; so a
    later chain object of the same id but another first conformer keeps
    none of them, as stock does.
    """
    if not raw:
        return []
    index = {k: i for i, k in enumerate(model.residues)}
    run_of = [int(model.run_of[index[(r[0], r[1])]]) for r in raw]
    chain_of = {}
    for (chain, _), run in zip(model.residues, model.run_of):
        chain_of.setdefault(int(run), chain)
    first, prev = {}, None
    for run in sorted(set(run_of)):
        if chain_of[run] != prev:
            start, prev = run, chain_of[run]
        first[run] = (model.conformer0 or {}).get(start, '')
    return [r[:8] for r, run in zip(raw, run_of)
            if r[6] is not None and not (r[8] != first[run] and r[9] == '')]


def count_omega(records):
    """As omegalyze counts: peptides by the residue after them, proline or not, and by type"""
    counts = dict.fromkeys(OMEGA_COUNTS, 0)
    for r in records:
        kind = 'proline' if r[2] == 'PRO' else 'general'
        counts[kind] += 1
        if r[7] != 'Trans':
            counts['%s_%s' % (r[7].lower(), kind)] += 1
    return counts


def protein_runs(counts):
    """
    The runs iotbx's chain.is_protein takes for protein, from their residue
    counts ({run: {'n', 'aa', 'na', 'unk'}}, residues.cablam's, summed over
    blocks): amino acids outnumber nucleotides and are at least 80% of the
    residues other than HOH, or all of those are UNK
    """
    return {run for run, c in counts.items()
            if c['n'] and (c['aa'] > c['na'] and c['aa'] / c['n'] >= 0.8 or c['unk'] == c['n'])}


def cablam_runs(model, classes):
    """residues.cablam's residue classes, from all blocks, summed by run: {run: {'n', 'aa', 'na', 'unk'}}"""
    index = {k: i for i, k in enumerate(model.residues)}
    counts = {}
    for chain, resid, *values in classes:
        c = counts.setdefault(int(model.run_of[index[(chain, resid)]]), {'n': 0, 'aa': 0, 'na': 0, 'unk': 0})
        for k, v in zip(('n', 'aa', 'na', 'unk'), values):
            c[k] += v
    return counts


def filter_cablam(model, records, protein):
    """
    The CaBLAM records cablamalyze keeps: those of protein chain objects
    (runs in protein), and of each chain id only the last such chain
    object, as it keeps a chain id's results in one place, which each
    chain object of that id takes in turn
    """
    if not records:
        return []
    index = {k: i for i, k in enumerate(model.residues)}
    chain_of = {}
    for (chain, _), run in zip(model.residues, model.run_of):
        chain_of.setdefault(int(run), chain)
    last = {}
    for run in sorted(protein):
        last[chain_of[run]] = run
    return [r for r in records if last.get(r[0]) == int(model.run_of[index[(r[0], r[1])]])]


def _resid_order(resid):
    m = re.match(r'(-?\d+)(.*)$', resid)
    return int(m.group(1)), m.group(2)


def count_cablam(records):
    """
    As cablamalyze's make_summary_stats counts, over its results in its
    order (chain id padded to two characters, residue number, insertion
    code, altloc): a residue is counted, and its altlocs' worst flags, by
    its first record that has a CaBLAM score; after the last residue its
    count is added once more, as stock does
    """
    counts = dict.fromkeys(('residues', 'ca_residues', 'outliers', 'disfavored', 'ca_outliers'), 0)
    order = sorted(records, key=lambda r: (r[0].rjust(2),) + _resid_order(r[1]) + (r[3],))
    prev, flags, is_residue = None, [0, 0, 0], 0
    for r in order:
        is_residue = 0 if r[4] is None else 1
        if (r[0], r[1]) != prev:
            counts['residues'] += is_residue
            counts['ca_residues'] += r[5] is not None
            for k, f in zip(('outliers', 'disfavored', 'ca_outliers'), flags):
                counts[k] += f
            flags = [0, 0, 0]
        if r[4] is not None and r[4] < CABLAM_OUTLIER and is_residue:
            flags[0] = 1
        if r[4] is not None and r[4] < CABLAM_DISFAVORED and is_residue:
            flags[1] = 1
        if r[5] is not None and r[5] < CA_GEOM_OUTLIER:
            flags[2] = 1
        prev = (r[0], r[1])
    counts['residues'] += is_residue
    for k, f in zip(('outliers', 'disfavored', 'ca_outliers'), flags):
        counts[k] += f
    return counts


def cablam_flagged(records):
    """Records of residues CaBLAM flags: disfavoured (which includes outliers) or CA geometry outliers"""
    return [r for r in records if r[4] is not None and r[4] < CABLAM_DISFAVORED or
            r[5] is not None and r[5] < CA_GEOM_OUTLIER]


def _atoms(outlier):
    n = 0
    while n < len(outlier) and isinstance(outlier[n], tuple):
        n += 1
    return outlier[:n]


def assemble(residues, resnames, ramachandran, rotamers, clashes, probe_atoms, geometry,
             residue_geometry, provenance, cbeta, omega, cablam, ramachandran_counts=None,
             rotamer_counts=None, cbeta_counts=None, omega_counts=None, cablam_counts=None, score=None):
    """
    A result from its parts, in canonical order: records by residue in file
    order then altloc, clashes by atom pair, outliers by atoms. residues:
    (chain, resid) in file order; geometry: {definition: {kind: {'total',
    'outliers'}}}, kinds as KINDS; residue_geometry: {(chain, resid): [nb,
    z2b, na, z2a]} for v34; cbeta, omega, cablam: every record
    (residues.cbeta, filter_omega, filter_cablam), of which the result keeps
    the outliers (C-beta), the peptides that are not trans, and the residues
    CaBLAM flags. Counts and the MolProbity score are
    computed unless given (stock's, whose records are its outliers only).
    """
    index = {k: i for i, k in enumerate(residues)}

    def by_residue(r):
        return (index[(r[0], r[1])], r[3], r)

    rama = sorted(ramachandran, key=by_residue)
    rota = sorted(rotamers, key=by_residue)
    clash = sorted(clashes, key=lambda c: (c[0], c[1], c[2], c[3]))
    geo = {d: {k: {'total': g[k]['total'],
                   'outliers': sorted(g[k]['outliers'], key=lambda o: (_atoms(o), o[len(_atoms(o)):]))}
               for k in KINDS[d]} for d, g in geometry.items()}
    rama_counts = ramachandran_counts or count_ramachandran(rama)
    rota_counts = rotamer_counts or count_rotamers(rota)
    clashscore = len(clash) * 1000 / probe_atoms if probe_atoms else 0.0
    entries = []
    for key, name in zip(residues, resnames):
        entry = {'chain': key[0], 'resid': key[1], 'resname': name}
        if 'v34' in geometry:
            nb, zb, na, za = residue_geometry.get(key, (0, 0.0, 0, 0.0))
            entry['v34'] = {'bonds': [nb, zb], 'angles': [na, za]}
        entries.append(entry)
    return {
        'schema': SCHEMA, 'version': VERSION, 'provenance': provenance,
        'ramachandran': dict(rama_counts, records=rama),
        'rotamers': dict(rota_counts, records=rota),
        'clashes': {'clashscore': clashscore, 'atoms': probe_atoms, 'records': clash},
        'molprobity_score': molprobity_score(clashscore, rota_counts, rama_counts) if score is None else score,
        'cbeta': dict(cbeta_counts or count_cbeta(cbeta), records=sorted(cbeta_outliers(cbeta), key=by_residue)),
        'omega': dict(omega_counts or count_omega(omega),
                      records=sorted((r for r in omega if r[7] != 'Trans'), key=by_residue)),
        'cablam': dict(cablam_counts or count_cablam(cablam), records=sorted(cablam_flagged(cablam), key=by_residue)),
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
        for r in br['ramachandran'] + br['rotamers'] + br['cbeta'] + br['omega'] + br['cablam']:
            _check_owned(owners, br, (r[0], r[1]), 'record')
        for c in br['clashes']:
            _check_owned(owners, br, (c[0][0], c[0][1]), 'clash')
        for d, g in br['geometry'].items():
            for k in KINDS[d]:
                for o in g[k]['outliers']:
                    _check_owned(owners, br, (o[0][0], o[0][1]), 'geometry outlier')
        for key in br['geometry'].get('v34', {}).get('residues', {}):
            _check_owned(owners, br, key, 'residue geometry')
    geometry = {d: {k: {'total': sum(br['geometry'][d][k]['total'] for br in block_results),
                        'outliers': [o for br in block_results for o in br['geometry'][d][k]['outliers']]}
                    for k in KINDS[d]} for d in definitions}
    residue_geometry = {}
    for br in block_results:
        for key, values in br['geometry'].get('v34', {}).get('residues', {}).items():
            residue_geometry[key] = values

    def every(kind):
        return [r for br in block_results for r in br[kind]]

    # stock's rules over the whole model: omegalyze's per chain id, CaBLAM's last protein chain object
    omega = filter_omega(model, every('omega'))
    cablam = filter_cablam(model, every('cablam'), protein_runs(cablam_runs(model, every('cablam_classes'))))
    return assemble(model.residues, model.resnames, every('ramachandran'), every('rotamers'), every('clashes'),
                    sum(br['probe_atoms'] for br in block_results), geometry, residue_geometry, provenance,
                    every('cbeta'), omega, cablam)


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
    for kind in ('ramachandran', 'rotamers', 'clashes', 'cbeta', 'omega', 'cablam'):
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
