"""
Differences between two results: identities and counts exact, values within
a tolerance; provenance (times, paths, versions) is not compared.

usage: remolprobity compare A B [--tol T] [--limit N] [--json]

A and B are results of remolprobity or of remolprobity stock, pickle or JSON
('-' reads JSON from stdin). Prints a summary of both, then each difference
by its place in the result. Exit status: 0 the same, 1 different, 2 a file
that is not a result.
"""
import argparse
import json
import math
import re
import sys

# the fields of records, to name them in differences
FIELDS = {'ramachandran': ('chain', 'resid', 'resname', 'altloc', 'class', 'phi', 'psi', 'score'),
          'rotamers': ('chain', 'resid', 'resname', 'altloc', 'evaluation', 'name', 'score'),
          'clashes': ('atom1', 'atom2', 'overlap', 'distance')}
OUTLIER_VALUES = {'v34': ('observed', 'ideal', 'sigma', 'z'), 'v33': ('observed', 'ideal', 'z')}
ATOM = ('chain', 'resid', 'resname', 'name', 'altloc')
RECORD = re.compile(r'\.(ramachandran|rotamers|clashes)\.records\[\d+\]$')
OUTLIER = re.compile(r'\.geometry\.(v3[34])\.(?:bonds|angles)\.outliers\[\d+\]$')
RECORD_ATOM = re.compile(r'\.(?:clashes\.records|geometry\.v3[34]\.(?:bonds|angles)\.outliers)\[\d+\]\.atom\d$')


def _identity(record):
    return tuple(x for x in record if not isinstance(x, float))


def _fields(path, n):
    """Names of a record's fields, if path is a record whose layout is known"""
    m = RECORD.search(path)
    if m and len(FIELDS[m.group(1)]) == n:
        return FIELDS[m.group(1)]
    m = OUTLIER.search(path)
    if m:
        values = OUTLIER_VALUES[m.group(1)]
        if n > len(values):
            return tuple('atom%d' % (i + 1) for i in range(n - len(values))) + values
    if RECORD_ATOM.search(path) and n == len(ATOM):
        return ATOM
    return None


def differences(a, b, tol=1e-9, limit=50, path='', out=None):
    """Where a and b differ, as readable lines (at most limit)"""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
                out.append('%s: %r != %r' % (path, a, b))
        else:
            out.append('%s: %r != %r' % (path, a, b))
        return out
    if type(a) is not type(b):
        out.append('%s: %s != %s' % (path, type(a).__name__, type(b).__name__))
        return out
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=repr):
            if k not in a or k not in b:
                out.append('%s.%s: %s only' % (path, k, 'first' if k in a else 'second'))
            else:
                differences(a[k], b[k], tol, limit, '%s.%s' % (path, k), out)
        return out
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            if all(isinstance(x, tuple) for x in list(a) + list(b)):
                ia, ib = {_identity(x) for x in a}, {_identity(x) for x in b}
                for k in sorted(ia - ib, key=repr)[:5]:
                    out.append('%s: %r first only' % (path, k))
                for k in sorted(ib - ia, key=repr)[:5]:
                    out.append('%s: %r second only' % (path, k))
            out.append('%s: %d != %d items' % (path, len(a), len(b)))
            return out
        names = _fields(path, len(a)) if isinstance(a, tuple) else None
        for i, (x, y) in enumerate(zip(a, b)):
            differences(x, y, tol, limit, '%s.%s' % (path, names[i]) if names else '%s[%d]' % (path, i), out)
        return out
    if a != b:
        out.append('%s: %r != %r' % (path, a, b))
    return out


def summary(r):
    """The headline numbers of a result"""
    p = r.get('provenance', {})
    s = {'made by': 'stock MolProbity' if p.get('stock') else 'remolprobity %s' % p.get('remolprobity', '?'),
         'blocks': p.get('blocks', {}).get('count'), 'residues': len(r['residues']),
         'Ramachandran': '%d, %d outliers' % (r['ramachandran']['total'], r['ramachandran']['outliers']),
         'rotamers': '%d, %d outliers' % (r['rotamers']['total'], r['rotamers']['outliers']),
         'clashscore': r['clashes']['clashscore'], 'clashes': len(r['clashes']['records'])}
    return s


def common(a, b):
    """
    a and b without provenance, and without a geometry definition only one
    of them computed (with the per-residue v34 sums); and what was left out
    """
    a = {k: v for k, v in a.items() if k != 'provenance'}
    b = {k: v for k, v in b.items() if k != 'provenance'}
    skipped = []
    for d in sorted(set(a['geometry']) ^ set(b['geometry'])):
        skipped.append('geometry %s is only in %s: not compared' % (d, 'the first' if d in a['geometry'] else 'the second'))
        for r in (a, b):
            r['geometry'] = {k: v for k, v in r['geometry'].items() if k != d}
            if d == 'v34':
                r['residues'] = [{k: v for k, v in e.items() if k != 'v34'} for e in r['residues']]
    return a, b, skipped


def _cell(v):
    if v is None:
        return '-'
    return '%.2f' % v if isinstance(v, float) else str(v)


def main(argv=None):
    from . import result

    p = argparse.ArgumentParser(prog='remolprobity compare', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--tol', type=float, default=1e-6,
                   help='relative and absolute tolerance for values (default 1e-6; one block against '
                        'blocks agree within 1e-9)')
    p.add_argument('--limit', type=int, default=50, help='differences to list (default 50)')
    p.add_argument('--json', action='store_true', help='print the comparison as JSON')
    args = p.parse_args(argv)
    try:
        ra, rb = result.read(args.a), result.read(args.b)
    except (OSError, ValueError) as e:
        print('error: %s' % e, file=sys.stderr)
        return 2
    a, b, skipped = common(ra, rb)
    found = differences(a, b, tol=args.tol, limit=args.limit + 1)
    more = len(found) > args.limit
    found = [d[1:] if d.startswith('.') else d for d in found[:args.limit]]
    sa, sb = summary(ra), summary(rb)
    if args.json:
        json.dump({'same': not found, 'differences': found, 'more': more, 'not_compared': skipped,
                   'summary': [sa, sb]}, sys.stdout, indent=1)
        print()
        return 1 if found else 0
    rows = list(sa) + [k for k in sb if k not in sa]
    width = max(len(k) for k in rows)
    ca = max([len(args.a)] + [len(_cell(sa.get(k))) for k in rows])
    print('%-*s  %-*s  %s' % (width, '', ca, args.a, args.b))
    for k in rows:
        print('%-*s  %-*s  %s' % (width, k, ca, _cell(sa.get(k)), _cell(sb.get(k))))
    print()
    if not found:
        print('same (values within %g)' % args.tol)
        return 0
    print('%s differences%s (values within %g):' % ('first %d' % len(found) if more else str(len(found)),
                                                    ', and more (raise --limit)' if more else '', args.tol))
    for line in found:
        print('  ' + line)
    return 1


if __name__ == '__main__':
    sys.exit(main())
