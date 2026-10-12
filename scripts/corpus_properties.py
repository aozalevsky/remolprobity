#!/usr/bin/env python3
"""The properties a corpus model is chosen for. usage: corpus_properties.py FILE"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remolprobity import io  # noqa: E402
from remolprobity.ids import residue_number  # noqa: E402

NUCLEIC = {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'}
METALS = {'FE', 'ZN', 'MG', 'CA', 'MN', 'CU', 'NI', 'CO', 'NA', 'K'}
WATER = {'HOH', 'WAT', 'DOD'}
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def _pick(c, *names):
    for n in names:
        if n in c:
            return c[n]
    return None


def atoms(path):
    """(altloc, resname, name, element, occupancy, het, chain, resid) per atom, in file order"""
    if io.file_format(path) == 'pdb':
        for line in io.pdb_lines(path):
            line = line.ljust(80)
            occ = line[54:60].strip()
            yield (line[16].strip(), line[17:20].strip(), line[12:16].strip(),
                   line[76:78].strip().upper(), float(occ) if occ else 0.0,
                   line.startswith('HETATM'), line[20:22].strip(), line[22:27].strip())
        return
    rows = io.cif_rows(path)
    c = {t: i for i, t in enumerate(next(rows))}
    ialt, iocc, iins = _pick(c, 'label_alt_id'), _pick(c, 'occupancy'), _pick(c, 'pdbx_PDB_ins_code')
    ichain, iseq = _pick(c, 'auth_asym_id', 'label_asym_id'), _pick(c, 'auth_seq_id', 'label_seq_id')
    icomp, iname = _pick(c, 'auth_comp_id', 'label_comp_id'), _pick(c, 'label_atom_id', 'auth_atom_id')
    ielement, igroup = _pick(c, 'type_symbol'), _pick(c, 'group_PDB')
    for r in rows:
        alt = r[ialt] if ialt is not None else '.'
        occ = r[iocc] if iocc is not None else '?'
        ins = r[iins] if iins is not None else '?'
        yield ('' if alt in ('.', '?') else alt, r[icomp], r[iname],
               r[ielement].upper() if ielement is not None else '',
               0.0 if occ in ('.', '?') else float(occ),
               igroup is not None and r[igroup] == 'HETATM', r[ichain],
               r[iseq] + ('' if ins in ('.', '?') else ins))


def properties(path):
    from scipy.spatial import cKDTree

    m = io.read_model(path, check=False)
    altlocs = metals = hydrogens = zero = max_resseq = 0
    insertion, ligands, nucleic, long_ids, sg = set(), set(), set(), set(), []
    for i, (alt, resname, name, element, occ, het, chain, resid) in enumerate(atoms(path)):
        altlocs += bool(alt)
        if resid[-1:] in LETTERS:
            insertion.add((chain, resid))
        if het and resname not in WATER:
            ligands.add(resname)
        metals += element in METALS
        if resname in NUCLEIC:
            nucleic.add((chain, resid))
        hydrogens += element in ('H', 'D')
        zero += occ == 0
        max_resseq = max(max_resseq, residue_number(resid.rstrip(LETTERS) or resid))
        if len(chain) > 2:
            long_ids.add(chain)
        if resname == 'CYS' and name == 'SG':
            sg.append(m.xyz[i])
    disulfides = len(cKDTree(np.array(sg)).query_pairs(2.5)) if len(sg) > 1 else 0
    return {'format': m.format, 'atoms': m.n_atoms, 'altlocs': altlocs,
            'insertion_codes': len(insertion), 'ligands': sorted(ligands), 'metals': metals,
            'nucleic_residues': len(nucleic), 'disulfides': disulfides, 'hydrogens': hydrogens,
            'zero_occupancy': zero, 'max_resseq': max_resseq, 'long_chain_ids': len(long_ids)}


if __name__ == '__main__':
    print(json.dumps(properties(sys.argv[1]), indent=1))
