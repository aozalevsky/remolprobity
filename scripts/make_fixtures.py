#!/usr/bin/env python3
"""
Fixtures for remolprobity's tests, made from public models (mmCIF in and out).

  excerpt SRC OUT --chains A,B [--residues FIRST-LAST] [--fill]
      the atoms of these chains (and residue numbers)
  subset SRC OUT --atoms N [--fill]
      whole chains in file order until at least N atoms
  split-chain SRC OUT --chain A --at N
      the chain's residues up to number N, then the other chains, then the
      rest of the chain: one chain id in two places
  renumber SRC OUT --chain A --add N
      the chain's residue numbers plus N
  zero-occupancy SRC OUT --chain B
      occupancy 0 for every atom of the chain
  tile SRC OUT --copies N [--spacing A] [--fill]
      copies side by side on a grid, chains renamed <chain>_<copy>

--fill writes occupancy 1.00 and B 0.00 where the file gives none (9A1N
gives neither; probe skips atoms without occupancy).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remolprobity import io  # noqa: E402
from remolprobity.ids import residue_number  # noqa: E402

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def _rows(src):
    """tags, {tag: column}, rows of the atom_site loop"""
    rows = io.cif_rows(src)
    tags = next(rows)
    return tags, {t: i for i, t in enumerate(tags)}, rows


def _chain_column(c):
    return c.get('auth_asym_id', c.get('label_asym_id'))


def _number(resid):
    return residue_number(resid.rstrip(LETTERS) or resid)


def _write(out, tags, rows, fill=False):
    c = {t: i for i, t in enumerate(tags)}
    with open(out, 'w') as f:
        f.write('data_fixture\nloop_\n' + ''.join('_atom_site.%s\n' % t for t in tags))
        for r in rows:
            if fill:
                for tag, value in (('occupancy', '1.00'), ('B_iso_or_equiv', '0.00')):
                    if tag in c and r[c[tag]] in ('.', '?'):
                        r[c[tag]] = value
            f.write(' '.join(io.cif_value(t) for t in r) + '\n')
        f.write('#\n')


def excerpt(src, out, chains, residues=None, fill=False):
    """Write the atoms of these chains (and residue numbers) to out."""
    m = io.read_model(src, check=False)
    keep = [c in chains and (residues is None or residues[0] <= _number(r) <= residues[1])
            for c, r in m.residues]
    tags, _, rows = _rows(src)
    _write(out, tags, (r for i, r in enumerate(rows) if keep[m.res_of[i]]), fill)


def subset(src, out, atoms, fill=False):
    """Write whole chains in file order until at least this many atoms."""
    m = io.read_model(src, check=False)
    per_chain = {}
    for (chain, _), n in zip(m.residues, m.atoms_per_residue()):
        per_chain[chain] = per_chain.get(chain, 0) + int(n)
    chosen, total = set(), 0
    for chain in per_chain:  # in file order
        if total >= atoms:
            break
        chosen.add(chain)
        total += per_chain[chain]
    excerpt(src, out, chosen, fill=fill)


def split_chain(src, out, chain, at):
    """Write the chain up to residue number at, the other chains, then the rest of the chain."""
    m = io.read_model(src, check=False)
    first = [k for k, (c, r) in enumerate(m.residues) if c == chain and _number(r) <= at]
    rest = [k for k, (c, r) in enumerate(m.residues) if c == chain and _number(r) > at]
    others = [k for k, (c, _) in enumerate(m.residues) if c != chain]
    tags, _, rows = _rows(src)
    by_residue = {}
    for i, r in enumerate(rows):
        by_residue.setdefault(int(m.res_of[i]), []).append(r)
    _write(out, tags, (r for k in first + others + rest for r in by_residue[k]))


def renumber(src, out, chain, add):
    """Write the model with the chain's residue numbers plus add."""
    tags, c, rows = _rows(src)
    ichain = _chain_column(c)

    def shifted(r):
        if r[ichain] == chain:
            for t in ('auth_seq_id', 'label_seq_id'):
                if t in c and r[c[t]].lstrip('-').isdigit():
                    r[c[t]] = str(int(r[c[t]]) + add)
        return r

    _write(out, tags, (shifted(r) for r in rows))


def zero_occupancy(src, out, chain):
    """Write the model with occupancy 0 for every atom of the chain."""
    tags, c, rows = _rows(src)
    ichain, iocc = _chain_column(c), c['occupancy']

    def zeroed(r):
        if r[ichain] == chain:
            r[iocc] = '0.00'
        return r

    _write(out, tags, (zeroed(r) for r in rows))


def tile(src, out, copies, spacing=10.0, fill=False):
    """Write copies of the model side by side on a grid, chains renamed <chain>_<copy>."""
    m = io.read_model(src, check=False)
    step = np.ptp(m.xyz, axis=0) + spacing
    side = 1
    while side ** 3 < copies:
        side += 1
    tags, c, _ = _rows(src)
    ixyz = [c['Cartn_x'], c['Cartn_y'], c['Cartn_z']]
    ichains = [c[t] for t in ('auth_asym_id', 'label_asym_id') if t in c]
    iserial = c.get('id')

    def copy(k):
        shift = step * np.array([k % side, k // side % side, k // side // side])
        _, _, rows = _rows(src)
        for i, r in enumerate(rows):
            for j, d in zip(ixyz, shift):
                r[j] = '%.3f' % (float(r[j]) + d)
            for j in ichains:
                r[j] = '%s_%d' % (r[j], k)
            if iserial is not None:
                r[iserial] = str(k * m.n_atoms + i + 1)
            yield r

    _write(out, tags, (r for k in range(copies) for r in copy(k)), fill)


def main(argv=None):
    """Command line: one subcommand per function above."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('excerpt', 'subset', 'split-chain', 'renumber', 'zero-occupancy', 'tile'):
        s = sub.add_parser(name)
        s.add_argument('src')
        s.add_argument('out')
        if name == 'excerpt':
            s.add_argument('--chains', required=True)
            s.add_argument('--residues')
        if name == 'subset':
            s.add_argument('--atoms', type=int, required=True)
        if name in ('split-chain', 'renumber', 'zero-occupancy'):
            s.add_argument('--chain', required=True)
        if name == 'split-chain':
            s.add_argument('--at', type=int, required=True)
        if name == 'renumber':
            s.add_argument('--add', type=int, required=True)
        if name == 'tile':
            s.add_argument('--copies', type=int, required=True)
            s.add_argument('--spacing', type=float, default=10.0)
        if name in ('excerpt', 'subset', 'tile'):
            s.add_argument('--fill', action='store_true')
    a = p.parse_args(argv)
    if a.cmd == 'excerpt':
        rng = tuple(int(x) for x in a.residues.split('-')) if a.residues else None
        excerpt(a.src, a.out, set(a.chains.split(',')), rng, a.fill)
    elif a.cmd == 'subset':
        subset(a.src, a.out, a.atoms, a.fill)
    elif a.cmd == 'split-chain':
        split_chain(a.src, a.out, a.chain, a.at)
    elif a.cmd == 'renumber':
        renumber(a.src, a.out, a.chain, a.add)
    elif a.cmd == 'zero-occupancy':
        zero_occupancy(a.src, a.out, a.chain)
    else:
        tile(a.src, a.out, a.copies, a.spacing, a.fill)


if __name__ == '__main__':
    main()
