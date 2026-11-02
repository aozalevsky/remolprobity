"""
A model as text: read_model reads its atoms once into arrays, naming
residues as iotbx's hierarchy does; write_blocks writes the atoms of sets of
residues, each to its own file, in a second pass. cctbx is not needed; iotbx's
polymer residue classes are used when it is importable.
"""
import gzip
import hashlib
import os
import re
from dataclasses import dataclass

import numpy as np

from .ids import residue_id

CHUNK = 1 << 20  # atoms per numpy chunk while reading


class InputError(Exception):
    """A model remolprobity can't take"""


def _open(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cif_tokens(line):
    """Tokens of one line of a CIF loop: plain, 'quoted' or "quoted" """
    if '"' not in line and "'" not in line:
        return line.split()
    out, i, n = [], 0, len(line)
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            break
        q = line[i]
        if q in '\'"':
            j = i + 1
            while True:  # a quote closes only before whitespace or the end
                j = line.find(q, j)
                if j == -1 or j + 1 >= n or line[j + 1].isspace():
                    break
                j += 1
            j = n if j == -1 else j
            out.append(line[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            out.append(line[i:j])
            i = j
    return out


_WS_RE = re.compile(r'\s')
_CIF_RESERVED = frozenset(('loop_', 'stop_', 'global_'))


def cif_value(tok):
    """tok written as a CIF value"""
    if tok and tok[0] not in '\'"_#$;[]' and not _WS_RE.search(tok):
        low = tok.lower()
        if low not in _CIF_RESERVED and not low.startswith(('data_', 'save_')):
            return tok
    return '"%s"' % tok if '"' not in tok else "'%s'" % tok


def file_format(path):
    """'cif' or 'pdb', from the first line that isn't blank or a comment"""
    with _open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                return 'cif' if s.startswith('data_') else 'pdb'
    raise InputError('%s is empty' % path)


def cif_rows(path, _pos=None):
    """The atom_site loop of an mmCIF file: its tags, then its rows as token lists"""
    tags = []
    track = _pos is not None and _pos[2]
    f_ctx = open(path, 'r', newline='') if track else _open(path)
    with f_ctx as f:
        it = iter(f)
        line = ''
        offset = 0
        line_start = 0
        for line in it:
            if track:
                line_start = offset
                offset += len(line)
            if line.startswith('_atom_site.'):
                tags.append(line.split()[0][len('_atom_site.'):])
            elif tags:
                break
        if not tags:
            raise InputError('%s has no atom_site loop' % path)
        yield tags
        n_tags = len(tags)
        row = []
        while line is not None:
            if line.startswith(';'):
                raise InputError('%s: text fields in atom_site are not supported' % path)
            if line.startswith(('#', 'loop_', '_', 'data_')):
                break
            tokens = cif_tokens(line)
            if not row and len(tokens) == n_tags:
                if track:
                    if not line.endswith('\n') or line.endswith('\r\n') or not line.isascii():
                        track = _pos[2] = False
                    else:
                        _pos[0] = line_start
                        _pos[1] = offset
                yield tokens
            else:
                if track:
                    track = _pos[2] = False
                row += tokens
                while len(row) >= n_tags:
                    yield row[:n_tags]
                    row = row[n_tags:]
            line = next(it, None)
            if line is not None and track:
                line_start = offset
                offset += len(line)
        if row:
            raise InputError('%s: atom_site ends in the middle of a row' % path)


@dataclass
class Model:
    """The atoms of a model as arrays, and its residues in file order"""
    path: str
    format: str
    sha256: str
    tags: list
    xyz: np.ndarray
    res_of: np.ndarray
    residues: list
    resnames: list
    position: np.ndarray
    chain_ids: list
    residue_names: list
    occupancy_max: float
    zero_occupancy: int
    hydrogens: int
    # index of the chain object iotbx puts each residue in, reading the whole file
    run_of: np.ndarray = None
    # the atoms through which cctbx links a polymer residue to the next (C, O3') and to the one before (N, P):
    # the first of each residue (nan where it has none), and, for residues with more (altlocs), all of them
    # {residue: (out, in)}
    link_out: np.ndarray = None
    link_in: np.ndarray = None
    link_more: dict = None
    # whether the residue has a CA (its position is then the CA's), and N, CA, C and O
    has_ca: np.ndarray = None
    mainchain: np.ndarray = None
    # {run: residues}: in each chain object with altlocs, the residues where each altloc first appears
    # (among all atoms, and among atoms other than hydrogens): iotbx orders a chain's conformers by that
    introducers: dict = None
    # {run: altloc}: the first conformer's altloc of each chain object with altlocs, without hydrogens
    conformer0: dict = None
    # (span_res, span_start, span_end) byte spans of contiguous residue runs in path, when single-line uncompressed
    spans: tuple = None

    @property
    def n_atoms(self):
        return len(self.xyz)

    def atoms_per_residue(self):
        return np.bincount(self.res_of, minlength=len(self.residues))


# atom names through which cctbx links polymer residues: 0, to the next residue (peptide C, nucleic O3');
# 1, to the one before (N, P)
LINK_SIDE = {'C': 0, "O3'": 0, 'O3*': 0, 'N': 1, 'P': 1}
_NAN3 = (np.nan, np.nan, np.nan)


class _Table:
    """Atoms as they are read, into numpy chunks"""

    def __init__(self, path=''):
        self.path = path
        self.xyz_chunks, self.res_chunks, self.xyz, self.res = [], [], [], []
        self.index, self.residues, self.resnames = {}, [], []
        self.ca_list, self.has_o_list = [], []
        self.chains, self.names, self.runs = {}, {}, []
        self.occupancy_max, self.zero_occupancy, self.hydrogens = 0.0, 0, 0
        # first altloc of each residue k, extra (k, resname, altloc) groups, and last state
        self.first_group_alt = []
        self.extra_groups = set()
        self.last = None
        self.last_group = None
        self.last_k = None
        self.last_polymer = False
        self.link_out, self.link_in, self.link_more = [], [], {}
        self.polymer = {}  # resname: whether iotbx links it in a polymer
        self.first_altloc = ({}, {})  # {run: {altloc: residue}}, among all atoms and among heavy atoms

    def add(self, chain, resseq, icode, resname, name, element, xyz, occupancy, run=0, altloc=''):
        cur_group = (chain, resseq, icode, resname, run, altloc)
        if cur_group == self.last_group:
            k = self.last_k
            polymer = self.last_polymer
            if altloc and element not in ('H', 'D') and run == self.runs[k]:
                self.first_altloc[1].setdefault(run, {}).setdefault(altloc, k)
        else:
            key = (chain, residue_id(resseq, icode))
            k = self.index.get(key)
            r_strip = resname.strip()
            if k is None:
                k = self.index[key] = len(self.residues)
                self.residues.append(key)
                self.runs.append(run)  # a residue is in the run of its first atom
                self.resnames.append(r_strip)
                self.first_group_alt.append(altloc)
                self.ca_list.append(None)
                self.has_o_list.append(False)
                self.link_out.append(_NAN3)
                self.link_in.append(_NAN3)
                self.chains.setdefault(chain, None)
                self.names.setdefault(resname, None)
            else:
                if not (r_strip == self.resnames[k] and altloc == self.first_group_alt[k]):
                    group = (k, r_strip, altloc)
                    if group not in self.extra_groups and self.last != (k, run):
                        # a residue seen before, again after other atoms or in another chain object:
                        # merge_atoms_at_end_to_residues, as stock runs it, puts an atom group back only into an earlier
                        # one of the same name and altloc; any other stays a residue of its own, which remolprobity,
                        # naming residues by chain and number, cannot tell apart
                        raise InputError("%s: residue %s %s appears again, in two chain objects or after other residues, as "
                                         "%s with altloc '%s', which it did not have before; iotbx keeps that as a residue of "
                                         'its own, which remolprobity cannot tell apart; give it a residue number of its own'
                                         % (self.path, chain, key[1], r_strip, altloc))
                    self.extra_groups.add(group)
                self.names.setdefault(resname, None)
            self.last = (k, run)
            self.last_group = cur_group
            self.last_k = k
            polymer = self.polymer.get(resname)
            if polymer is None:
                polymer = self.polymer[resname] = is_polymer_residue(resname)
            self.last_polymer = polymer
            if altloc and run == self.runs[k]:
                self.first_altloc[0].setdefault(run, {}).setdefault(altloc, k)
                if element not in ('H', 'D'):
                    self.first_altloc[1].setdefault(run, {}).setdefault(altloc, k)
        self.res.append(k)
        self.xyz.append(xyz)
        if name == 'CA' and self.ca_list[k] is None:
            self.ca_list[k] = xyz
        elif name == 'O':
            self.has_o_list[k] = True
        side = LINK_SIDE.get(name)
        if side is not None and polymer:
            first_list = self.link_out if side == 0 else self.link_in
            if first_list[k] is _NAN3:
                first_list[k] = xyz
                if k in self.link_more:
                    self.link_more[k][side].append(xyz)
            else:
                pair = self.link_more.get(k)
                if pair is None:
                    pair = self.link_more[k] = (
                        [self.link_out[k]] if self.link_out[k] is not _NAN3 else [],
                        [self.link_in[k]] if self.link_in[k] is not _NAN3 else [],
                    )
                pair[side].append(xyz)
        if not occupancy:
            self.zero_occupancy += 1
        elif occupancy > self.occupancy_max:
            self.occupancy_max = occupancy
        if element in ('H', 'D'):
            self.hydrogens += 1
        if len(self.res) >= CHUNK:
            self.flush()
        return k

    def flush(self):
        if self.res:
            self.xyz_chunks.append(np.array(self.xyz, dtype=np.float64))
            self.res_chunks.append(np.array(self.res, dtype=np.int64))
            self.xyz, self.res = [], []

    def model(self, path, fmt, tags, spans=None):
        self.flush()
        if not self.residues:
            raise InputError('%s has no atoms' % path)
        xyz = np.concatenate(self.xyz_chunks)
        res_of = np.concatenate(self.res_chunks)
        counts = np.bincount(res_of, minlength=len(self.residues))
        position = np.zeros((len(self.residues), 3))
        np.add.at(position, res_of, xyz)
        position /= counts[:, None]
        for k, p in enumerate(self.ca_list):
            if p is not None:
                position[k] = p
        has_ca = np.array([p is not None for p in self.ca_list], dtype=bool)
        has_o = np.array(self.has_o_list, dtype=bool)
        link_out = np.array(self.link_out, dtype=np.float64).reshape(len(self.residues), 3)
        link_in = np.array(self.link_in, dtype=np.float64).reshape(len(self.residues), 3)
        mainchain = has_ca & has_o & ~np.isnan(link_out[:, 0]) & ~np.isnan(link_in[:, 0])
        introducers = {}
        for found in self.first_altloc:
            for run, first in found.items():
                introducers.setdefault(run, set()).update(first.values())
        return Model(path=path, format=fmt, sha256=sha256(path), tags=tags, xyz=xyz,
                     res_of=res_of, residues=self.residues, resnames=self.resnames,
                     position=position, chain_ids=list(self.chains),
                     residue_names=list(self.names), occupancy_max=self.occupancy_max,
                     zero_occupancy=self.zero_occupancy, hydrogens=self.hydrogens,
                     run_of=np.array(self.runs, dtype=np.int64), link_out=link_out, link_in=link_in,
                     link_more=self.link_more, has_ca=has_ca, mainchain=mainchain,
                     introducers={run: sorted(ks) for run, ks in introducers.items()},
                     conformer0={run: next(iter(first)) for run, first in self.first_altloc[1].items()},
                     spans=spans)


def _columns(path, tags):
    """The columns iotbx's hierarchy takes each item from (iotbx/pdb/mmcif.py)"""
    col = {t: i for i, t in enumerate(tags)}

    def pick(*names):
        for n in names:
            if n in col:
                return col[n]
        return None

    c = {'chain': pick('auth_asym_id', 'label_asym_id'),
         'seq': pick('auth_seq_id', 'label_seq_id'),
         'ins': pick('pdbx_PDB_ins_code'),
         'comp': pick('auth_comp_id', 'label_comp_id'),
         'atom': pick('label_atom_id', 'auth_atom_id'),
         'element': pick('type_symbol'),
         'x': pick('Cartn_x'), 'y': pick('Cartn_y'), 'z': pick('Cartn_z'),
         'occ': pick('occupancy'), 'model': pick('pdbx_PDB_model_num'), 'alt': pick('label_alt_id')}
    missing = [k for k in ('chain', 'seq', 'comp', 'atom', 'x', 'y', 'z') if c[k] is None]
    if missing:
        raise InputError('%s: atom_site has no %s' % (path, ', '.join(missing)))
    return c


_POLYMER = frozenset('''ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR
VAL MSE A C G U DA DC DG DT'''.split())


def is_polymer_residue(resname):
    """
    Whether iotbx's mmCIF reader counts the residue as amino acid or RNA/DNA,
    which decides if a change of label_asym_id after it starts a chain. Uses
    iotbx's own table where cctbx is importable, else the common residues.
    """
    resname = resname.strip()
    try:
        from iotbx.pdb import common_residue_names_get_class
    except ImportError:
        return resname in _POLYMER
    return common_residue_names_get_class(resname) in (
        'common_amino_acid', 'modified_amino_acid', 'common_rna_dna', 'modified_rna_dna')


def _starts_chain(auth, label, prev_auth, prev_label, prev_comp):
    """iotbx/pdb/mmcif.py: a new chain object when the auth chain changes, or the label chain changes after a polymer atom"""
    return auth != prev_auth or (label != prev_label and is_polymer_residue(prev_comp))


# atom_site items iotbx reads as numbers: '.' or '?' there, or no such item, and it cannot read the file
VALUE_COLUMNS = ('Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv')
# atom_site items iotbx's mmCIF reader cannot do without (besides those of
# _columns, which each have an alternative)
REQUIRED_COLUMNS = ('id', 'type_symbol', 'label_alt_id', 'label_entity_id', 'pdbx_PDB_model_num') + VALUE_COLUMNS
# the options that fill in a missing value, by atom_site item
FILL_OPTIONS = {'occupancy': '--zero-occupancy-as-one', 'B_iso_or_equiv': '--missing-b-as-zero'}


def _unreadable(path, atom, missing):
    """The refusal of an atom missing these values ((item, value)); it names the options, if they fill them all"""
    options = [FILL_OPTIONS[tag] for tag, _ in missing if tag in FILL_OPTIONS]
    return InputError("%s: atom %s has %s, which iotbx (and so stock MolProbity) cannot read%s" % (
        path, atom, ' and '.join("'%s' in _atom_site.%s" % (v, tag) for tag, v in missing),
        '; give %s, or use %s' % ('it' if len(missing) == 1 else 'them', ' '.join(options))
        if len(options) == len(missing) else ''))


def _read_cif(path, check=True, zero_occupancy_as_one=False, missing_b_as_zero=False):
    pos = [0, 0, not path.endswith('.gz')]
    rows = cif_rows(path, _pos=pos)
    tags = next(rows)
    c = _columns(path, tags)
    if check:
        for tag in REQUIRED_COLUMNS:
            if tag not in tags:
                raise InputError('%s: atom_site has no %s, without which iotbx (and so stock MolProbity) '
                                 'cannot read the model' % (path, tag))
        # the blocks get 1.00 where occupancy is missing with --zero-occupancy-as-one, and 0.00 where B
        # is with --missing-b-as-zero
        values = [(tags.index(t), t) for t in VALUE_COLUMNS
                  if not (zero_occupancy_as_one and t == 'occupancy' or missing_b_as_zero and t == 'B_iso_or_equiv')]
    t = _Table(path)
    model = None
    run = -1
    prev_chain = prev_label = prev_comp = None
    ilabel = tags.index('label_asym_id') if 'label_asym_id' in tags else c['chain']
    c_model, c_chain, c_seq, c_ins, c_comp = c['model'], c['chain'], c['seq'], c['ins'], c['comp']
    c_atom, c_elem, c_x, c_y, c_z, c_occ, c_alt = (
        c['atom'], c['element'], c['x'], c['y'], c['z'], c['occ'], c['alt']
    )
    span_res, span_run, span_lbl_idx, span_start, span_end = [], [], [], [], []
    lbl_map = {}
    span_k = -1
    span_r = -1
    span_lbl = None
    span_s = span_e = 0
    for r in rows:
        if c_model is not None:
            if model is None:
                model = r[c_model]
            elif r[c_model] != model:
                raise InputError('%s has more than one model; split them' % path)
        chain, seq = r[c_chain], r[c_seq]
        if chain in ('.', '?'):
            raise InputError('%s: atom %s has no chain id' % (path, ' '.join(r[:4])))
        if seq in ('.', '?'):
            raise InputError('%s: atom %s has no residue number (%s)'
                             % (path, ' '.join(r[:4]), tags[c_seq]))
        if check:
            for i, _ in values:
                if r[i] in ('.', '?'):
                    raise _unreadable(path, '%s (residue %s %s %s)' % (' '.join(r[:4]), chain, r[c_comp], seq),
                                      [(tag, r[j]) for j, tag in values if r[j] in ('.', '?')])
        ins = r[c_ins] if c_ins is not None else '?'
        occ = r[c_occ] if c_occ is not None else '?'
        alt = r[c_alt] if c_alt is not None else '.'
        label = r[ilabel]
        comp = r[c_comp]
        if chain != prev_chain or label != prev_label:
            if prev_chain is None or chain != prev_chain or is_polymer_residue(prev_comp):
                run += 1
            prev_chain, prev_label = chain, label
        prev_comp = comp
        k = t.add(chain, seq, '' if ins in ('.', '?') else ins, '%3s' % comp,
                  r[c_atom], r[c_elem].upper() if c_elem is not None else '',
                  (float(r[c_x]), float(r[c_y]), float(r[c_z])),
                  None if occ in ('.', '?') else float(occ), run, '' if alt in ('.', '?') else alt)
        if pos[2]:
            r_k = t.runs[k]
            if k != span_k or r_k != span_r or label != span_lbl:
                if span_k >= 0:
                    span_res.append(span_k)
                    span_run.append(span_r)
                    span_lbl_idx.append(lbl_map.setdefault(span_lbl, len(lbl_map)))
                    span_start.append(span_s)
                    span_end.append(span_e)
                span_k, span_r, span_lbl, span_s = k, r_k, label, pos[0]
            span_e = pos[1]
    spans = None
    if pos[2] and span_k >= 0:
        span_res.append(span_k)
        span_run.append(span_r)
        span_lbl_idx.append(lbl_map.setdefault(span_lbl, len(lbl_map)))
        span_start.append(span_s)
        span_end.append(span_e)
        spans = (
            np.array(span_res, dtype=np.int32),
            np.array(span_run, dtype=np.int32),
            np.array(span_lbl_idx, dtype=np.int32),
            np.array(span_start, dtype=np.int64),
            np.array(span_end, dtype=np.int64),
            list(lbl_map),
        )
    return t.model(path, 'cif', tags, spans=spans)


def read_model(path, check=True, zero_occupancy_as_one=False, missing_b_as_zero=False):
    """
    The model in an mmCIF or PDB file (optionally gzipped); one model only.
    With check, refuses, naming the atom and the item or columns, what
    iotbx and so stock MolProbity cannot read. mmCIF: '.' or '?' in
    coordinates, occupancy (unless zero_occupancy_as_one, which fills it)
    or B (unless missing_b_as_zero, which fills it), and an atom_site
    without an item iotbx needs (REQUIRED_COLUMNS).
    PDB: coordinates, occupancy or B that are not numbers as iotbx reads
    them (pdb_number; blank is 0, as for iotbx). check=False is for
    scripts that measure or rewrite models; they still need coordinates.
    """
    if file_format(path) == 'cif':
        return _read_cif(path, check, zero_occupancy_as_one, missing_b_as_zero)
    return _read_pdb(path, check)


def stretches(model):
    """
    One int per residue: the stretch it is in, a maximal run of consecutive
    residues (file order) with one chain id. reduce takes a stretch for one
    chain: it ignores TER and label_asym_id and sees only chain id changes.
    """
    chain = np.array([c for c, _ in model.residues], dtype=object)
    start = np.zeros(len(chain), dtype=np.int64)
    start[1:] = chain[1:] != chain[:-1]
    return np.cumsum(start)


def _pdb_records(path, _pos=None):
    """ATOM, HETATM and TER lines of a PDB file; refuses a second MODEL"""
    models = 0
    track = _pos is not None and _pos[2]
    f_ctx = open(path, 'r', newline='') if track else _open(path)
    offset = 0
    with f_ctx as f:
        for line in f:
            if track:
                line_start = offset
                offset += len(line)
            if line.startswith('MODEL'):
                models += 1
                if models > 1:
                    raise InputError('%s has more than one model; split them' % path)
            elif line.startswith(('ATOM  ', 'HETATM', 'TER')):
                if track:
                    if not line.endswith('\n') or line.endswith('\r\n') or not line.isascii():
                        track = _pos[2] = False
                    else:
                        _pos[0] = line_start
                        _pos[1] = offset
                yield line.rstrip('\r\n') if track else line.rstrip('\n')


def pdb_lines(path):
    """ATOM and HETATM lines of a PDB file; refuses a second MODEL"""
    return (line for line in _pdb_records(path) if not line.startswith('TER'))


_PDB_NUMBER = re.compile(r'[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?')
# the number columns of an atom record, as iotbx reads them (0-based, end excluded)
PDB_NUMBERS = (('x', 30, 38), ('y', 38, 46), ('z', 46, 54), ('occupancy', 54, 60), ('B', 60, 66))


def pdb_number(field):
    """
    A number column of a PDB atom record as iotbx reads it: leading blanks
    ignored, every later blank a 0 (as Fortran's BZ: '  1 0' is 100,
    '12  ' is 1200), all blank 0; None when iotbx cannot read it ('.', '?',
    'nan', ...)
    """
    s = field.lstrip(' ').replace(' ', '0')
    if not s:
        return 0.0
    return float(s) if _PDB_NUMBER.fullmatch(s) else None


def _read_pdb(path, check=True):
    t = _Table(path)
    run, chain, ended = 0, None, False
    pos = [0, 0, not path.endswith('.gz')]
    span_res, span_run, span_start, span_end = [], [], [], []
    span_k = -1
    span_r = -1
    span_s = span_e = 0
    for line in _pdb_records(path, _pos=pos):
        if line.startswith('TER'):
            ended = True  # iotbx starts a chain object after a TER
            if pos[2] and span_k >= 0:
                span_res.append(span_k)
                span_run.append(span_r)
                span_start.append(span_s)
                span_end.append(span_e)
                span_k = -1
            continue
        line = line.ljust(80)
        c_str = line[20:22].strip()
        if chain is not None and (ended or c_str != chain):
            run += 1
        chain, ended = c_str, False
        values = {}
        for column, i, j in PDB_NUMBERS:
            v = values[column] = pdb_number(line[i:j])
            if v is None and (check or column in ('x', 'y', 'z')):
                raise InputError("%s: atom %s has '%s' in columns %d-%d (%s), which iotbx (and so stock "
                                 "MolProbity) cannot read" % (path, line[:27].strip(), line[i:j].strip(),
                                                                 i + 1, j, column))
        name = line[12:16].strip()
        element = line[76:78].strip().upper() or name[:1].upper()
        k = t.add(c_str, line[22:26], line[26], line[17:20], name, element,
                  (values['x'], values['y'], values['z']), values['occupancy'], run, line[16].strip())
        if pos[2]:
            r_k = t.runs[k]
            if k != span_k or r_k != span_r:
                if span_k >= 0:
                    span_res.append(span_k)
                    span_run.append(span_r)
                    span_start.append(span_s)
                    span_end.append(span_e)
                span_k, span_r, span_s = k, r_k, pos[0]
            span_e = pos[1]
    spans = None
    if pos[2] and span_k >= 0:
        span_res.append(span_k)
        span_run.append(span_r)
        span_start.append(span_s)
        span_end.append(span_e)
    if pos[2] and span_res:
        spans = (
            np.array(span_res, dtype=np.int32),
            np.array(span_run, dtype=np.int32),
            None,
            np.array(span_start, dtype=np.int64),
            np.array(span_end, dtype=np.int64),
            None,
        )
    return t.model(path, 'pdb', [], spans=spans)


def _raise_nofile():
    """Raise the soft open-file limit to the hard one; quietly keep it if that fails"""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (ImportError, ValueError, OSError):
        pass


def _nofile_budget():
    """Files that can be open at once, leaving headroom for everything else"""
    _raise_nofile()
    try:
        import resource
        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if soft == resource.RLIM_INFINITY:
            return 1 << 20
    except (ImportError, ValueError, OSError):
        return 1000
    return max(1, soft - 64)


class _Segments:
    """
    Keeps one block file's chain objects as the model's runs: tells, for the
    next atom written, what label_asym_id to give it (cif) or whether a TER
    goes first (pdb), so that iotbx breaks the chain exactly where the model's
    runs do, and nowhere else.
    """

    def __init__(self, path):
        self.path = path
        self.run = self.auth = self.comp = None
        self.label_of = None

    def cif_label(self, run, auth, label, comp, label_is_chain, has_label):
        """
        The label to write for an atom of the given run; its own unless a
        break must be forced. Where iotbx will not split (after a non-polymer
        atom) two non-polymer runs are left merged, which changes no result;
        a polymer residue that would join the run before it is refused.
        """
        if self.run is None or run == self.run:
            if self.run is None:
                self.label_of = label
            out = self.label_of
        else:
            natural = _starts_chain(auth, label, self.auth, self.label_of, self.comp)
            out = label
            if not natural:
                if is_polymer_residue(self.comp):
                    if label_is_chain or not has_label:
                        raise InputError(
                            '%s: cannot keep the chain break before atoms of chain %s (run %d): the chain '
                            'id comes from label_asym_id, which cannot change' % (self.path, auth, run))
                    out = '%s~%d' % (label, run)
                elif is_polymer_residue(comp):
                    raise InputError(
                        '%s: cannot keep the chain break before a polymer residue of chain %s (run %d): '
                        'iotbx joins a chain only after a non-polymer atom, and no label_asym_id change '
                        'can split them' % (self.path, auth, run))
            self.label_of = out
        self.run, self.auth, self.comp = run, auth, comp
        return out

    def pdb_ter(self, run, chain):
        """Whether a TER must precede an atom of the given run"""
        ter = self.run is not None and run != self.run and chain == self.auth
        self.run, self.auth = run, chain
        return ter


def _write_batch(model, member_of, paths, batch, zero_occupancy_as_one, missing_b_as_zero):
    """One pass over the atoms, writing the blocks in batch (block indices) only"""
    local = {b: f for f, b in enumerate(batch)}
    files = []
    try:
        for b in batch:
            files.append(open(paths[b], 'w'))
        res_of = model.res_of
        run_of = model.run_of
        if model.format == 'cif':
            head = 'data_block\nloop_\n' + ''.join('_atom_site.%s\n' % t for t in model.tags)
            for f in files:
                f.write(head)
            rows = cif_rows(model.path)
            next(rows)
            iocc = model.tags.index('occupancy') if 'occupancy' in model.tags else None
            ib = model.tags.index('B_iso_or_equiv') if missing_b_as_zero and 'B_iso_or_equiv' in model.tags \
                else None
            cols = _columns(model.path, model.tags)
            c_chain, c_comp = cols['chain'], cols['comp']
            ilabel = model.tags.index('label_asym_id') if 'label_asym_id' in model.tags else None
            label_is_chain = c_chain == ilabel
            segments = [_Segments(paths[b]) for b in batch] if run_of is not None else None
            for i, r in enumerate(rows):
                res_i = res_of[i]
                bs = member_of[res_i]
                if not bs:
                    continue
                if zero_occupancy_as_one and iocc is not None and \
                        (r[iocc] in ('.', '?') or float(r[iocc]) == 0):
                    r[iocc] = '1.00'
                if ib is not None and r[ib] in ('.', '?'):
                    r[ib] = '0.00'
                base_line = None
                run_i = int(run_of[res_i]) if segments is not None else 0
                auth_i = r[c_chain]
                comp_i = r[c_comp]
                lbl_i = r[ilabel] if ilabel is not None else auth_i
                for b in bs:
                    loc = local.get(b)
                    if loc is not None:
                        if segments is not None:
                            out = segments[loc].cif_label(
                                run_i, auth_i, lbl_i, comp_i, label_is_chain, ilabel is not None)
                            if ilabel is not None and out != lbl_i:
                                row = list(r)
                                row[ilabel] = out
                                files[loc].write(' '.join(cif_value(t) for t in row) + '\n')
                                continue
                        if base_line is None:
                            base_line = ' '.join(cif_value(t) for t in r) + '\n'
                        files[loc].write(base_line)
            for f in files:
                f.write('#\n')
        else:
            segments = [_Segments(paths[b]) for b in batch] if run_of is not None else None
            for i, line in enumerate(pdb_lines(model.path)):
                res_i = res_of[i]
                bs = member_of[res_i]
                if not bs:
                    continue
                if zero_occupancy_as_one:
                    if not pdb_number(line.ljust(60)[54:60]):  # blank or 0, as iotbx reads it
                        line = line.ljust(60)[:54] + '%6.2f' % 1.0 + line[60:]
                out_line = line + '\n'
                run_i = int(run_of[res_i]) if segments is not None else 0
                chain_i = line[20:22].strip() if segments is not None else ''
                for b in bs:
                    loc = local.get(b)
                    if loc is not None:
                        if segments is not None and segments[loc].pdb_ter(run_i, chain_i):
                            files[loc].write('TER\n')
                        files[loc].write(out_line)
            for f in files:
                f.write('END\n')
    finally:
        for f in files:
            f.close()


def check_segments(model, members, paths=None):
    """
    Verify that every block in members can keep the model's chain breaks
    (_Segments.cif_label) without writing block files to disk.
    """
    if model.format != 'cif' or model.run_of is None:
        return
    if paths is None:
        paths = ['block%05d.cif' % b for b in range(len(members))]
    cols = _columns(model.path, model.tags)
    has_label = 'label_asym_id' in model.tags
    ilabel = model.tags.index('label_asym_id') if has_label else None
    label_is_chain = cols['chain'] == ilabel
    if model.spans is not None:
        span_res, span_run, span_lbl_idx, _, _, label_table = model.spans
        res_auth = [c for c, _ in model.residues]
        res_comp = [n.strip() for n in model.resnames]
        n_res = len(model.residues)
        for b, mem in enumerate(members):
            if len(mem) == 0:
                continue
            in_b = np.zeros(n_res, dtype=bool)
            in_b[np.asarray(mem, dtype=np.int64)] = True
            idx = np.flatnonzero(in_b[span_res])
            if len(idx) == 0:
                continue
            r_sub = span_run[idx]
            change = np.flatnonzero(r_sub[1:] != r_sub[:-1])
            firsts = idx[np.concatenate(([0], change + 1))]
            lasts = idx[np.concatenate((change, [len(idx) - 1]))]
            seg = _Segments(paths[b])
            for s_f, s_l in zip(firsts, lasts):
                rf = int(span_res[s_f])
                rl = int(span_res[s_l])
                auth_f = res_auth[rf]
                lbl_f = label_table[int(span_lbl_idx[s_f])] if has_label else auth_f
                seg.cif_label(int(span_run[s_f]), auth_f, lbl_f, res_comp[rf], label_is_chain, has_label)
                seg.auth = res_auth[rl]
                seg.comp = res_comp[rl]
        return
    member_of = [[] for _ in model.residues]
    for b, residues in enumerate(members):
        for r in residues:
            member_of[int(r)].append(b)
    segments = [_Segments(paths[b]) for b in range(len(members))]
    rows = cif_rows(model.path)
    next(rows)
    res_of = model.res_of
    run_of = model.run_of
    c_chain, c_comp = cols['chain'], cols['comp']
    for i, r in enumerate(rows):
        res_i = res_of[i]
        bs = member_of[res_i]
        if not bs:
            continue
        run_i = int(run_of[res_i])
        auth_i = r[c_chain]
        comp_i = r[c_comp]
        lbl_i = r[ilabel] if has_label else auth_i
        for b in bs:
            segments[b].cif_label(run_i, auth_i, lbl_i, comp_i, label_is_chain, has_label)


def model_slice(model, zero_occupancy_as_one=False, missing_b_as_zero=False):
    """Compact specification for extracting block text in memory inside workers"""
    auth_table = list(model.chain_ids)
    auth_map = {c: i for i, c in enumerate(auth_table)}
    res_auth_idx = np.array([auth_map[c] for c, _ in model.residues], dtype=np.int32)
    comp_table = sorted({n.strip() for n in model.resnames})
    comp_map = {c: i for i, c in enumerate(comp_table)}
    res_comp_idx = np.array([comp_map[n.strip()] for n in model.resnames], dtype=np.int32)
    spans = model.spans
    return {
        'path': model.path,
        'format': model.format,
        'tags': model.tags,
        'n_residues': len(model.residues),
        'auth_table': auth_table,
        'res_auth_idx': res_auth_idx,
        'comp_table': comp_table,
        'res_comp_idx': res_comp_idx,
        'spans': spans,
        'res_of': model.res_of if spans is None else None,
        'run_of': model.run_of if spans is None else None,
        'zero_occupancy_as_one': bool(zero_occupancy_as_one),
        'missing_b_as_zero': bool(missing_b_as_zero),
    }


def block_text(spec, members_b, path='<block>'):
    """
    Return the mmCIF or PDB text of a single block (members_b: residue indices)
    in memory, preserving exact _Segments chain breaks and occupancy/B fills.
    """
    n_res = spec['n_residues']
    in_b = np.zeros(n_res, dtype=bool)
    if len(members_b):
        in_b[np.asarray(members_b, dtype=np.int64)] = True
    spans = spec['spans']
    fmt = spec['format']
    zero_occ = spec['zero_occupancy_as_one']
    miss_b = spec['missing_b_as_zero']
    if spans is not None:
        span_res, span_run, span_lbl_idx, span_start, span_end, label_table = spans
        auth_table, res_auth_idx = spec['auth_table'], spec['res_auth_idx']
        comp_table, res_comp_idx = spec['comp_table'], spec['res_comp_idx']
        idx = np.flatnonzero(in_b[span_res])
        if fmt == 'cif':
            tags = spec['tags']
            head = 'data_block\nloop_\n' + ''.join('_atom_site.%s\n' % t for t in tags)
            if len(idx) == 0:
                return head + '#\n'
            cols = _columns(spec['path'], tags)
            has_label = 'label_asym_id' in tags
            ilabel = tags.index('label_asym_id') if has_label else None
            label_is_chain = cols['chain'] == ilabel
            iocc = tags.index('occupancy') if 'occupancy' in tags else None
            ib = tags.index('B_iso_or_equiv') if miss_b and 'B_iso_or_equiv' in tags else None
            r_sub = span_run[idx]
            change = np.flatnonzero(r_sub[1:] != r_sub[:-1])
            run_starts = np.concatenate(([0], change + 1))
            run_ends = np.concatenate((change + 1, [len(idx)]))
            seg = _Segments(path)
            chunks = [head]
            cur_start = cur_end = -1
            with open(spec['path'], 'r', newline='') as f:
                for m_s, m_e in zip(run_starts, run_ends):
                    s_f = int(idx[m_s])
                    s_l = int(idx[m_e - 1])
                    rf = int(span_res[s_f])
                    rl = int(span_res[s_l])
                    auth_f = auth_table[int(res_auth_idx[rf])]
                    lbl_f = label_table[int(span_lbl_idx[s_f])] if has_label else auth_f
                    out = seg.cif_label(
                        int(span_run[s_f]), auth_f, lbl_f, comp_table[int(res_comp_idx[rf])],
                        label_is_chain, has_label)
                    seg.auth = auth_table[int(res_auth_idx[rl])]
                    seg.comp = comp_table[int(res_comp_idx[rl])]
                    for m in range(int(m_s), int(m_e)):
                        s = int(idx[m])
                        st, en = int(span_start[s]), int(span_end[s])
                        lbl_s = label_table[int(span_lbl_idx[s])] if has_label else None
                        rewrite = (zero_occ and iocc is not None) or (ib is not None) or (has_label and out != lbl_s)
                        if not rewrite:
                            if cur_end == st:
                                cur_end = en
                            else:
                                if cur_start >= 0:
                                    f.seek(cur_start)
                                    chunks.append(f.read(cur_end - cur_start))
                                cur_start, cur_end = st, en
                        else:
                            if cur_start >= 0:
                                f.seek(cur_start)
                                chunks.append(f.read(cur_end - cur_start))
                                cur_start = cur_end = -1
                            f.seek(st)
                            raw = f.read(en - st)
                            for line in raw.splitlines():
                                row = cif_tokens(line)
                                if zero_occ and iocc is not None and (row[iocc] in ('.', '?') or float(row[iocc]) == 0):
                                    row[iocc] = '1.00'
                                if ib is not None and row[ib] in ('.', '?'):
                                    row[ib] = '0.00'
                                if has_label:
                                    row[ilabel] = out
                                chunks.append(' '.join(cif_value(t) for t in row) + '\n')
                if cur_start >= 0:
                    f.seek(cur_start)
                    chunks.append(f.read(cur_end - cur_start))
            chunks.append('#\n')
            return ''.join(chunks)
        else:
            if len(idx) == 0:
                return 'END\n'
            r_sub = span_run[idx]
            change = np.flatnonzero(r_sub[1:] != r_sub[:-1])
            run_starts = np.concatenate(([0], change + 1))
            run_ends = np.concatenate((change + 1, [len(idx)]))
            seg = _Segments(path)
            chunks = []
            cur_start = cur_end = -1
            with open(spec['path'], 'r', newline='') as f:
                for m_s, m_e in zip(run_starts, run_ends):
                    s_f = int(idx[m_s])
                    s_l = int(idx[m_e - 1])
                    rf = int(span_res[s_f])
                    rl = int(span_res[s_l])
                    auth_f = auth_table[int(res_auth_idx[rf])]
                    if seg.pdb_ter(int(span_run[s_f]), auth_f):
                        if cur_start >= 0:
                            f.seek(cur_start)
                            chunks.append(f.read(cur_end - cur_start))
                            cur_start = cur_end = -1
                        chunks.append('TER\n')
                    seg.auth = auth_table[int(res_auth_idx[rl])]
                    for m in range(int(m_s), int(m_e)):
                        s = int(idx[m])
                        st, en = int(span_start[s]), int(span_end[s])
                        if not zero_occ:
                            if cur_end == st:
                                cur_end = en
                            else:
                                if cur_start >= 0:
                                    f.seek(cur_start)
                                    chunks.append(f.read(cur_end - cur_start))
                                cur_start, cur_end = st, en
                        else:
                            if cur_start >= 0:
                                f.seek(cur_start)
                                chunks.append(f.read(cur_end - cur_start))
                                cur_start = cur_end = -1
                            f.seek(st)
                            raw = f.read(en - st)
                            for line in raw.splitlines():
                                if not pdb_number(line.ljust(60)[54:60]):
                                    line = line.ljust(60)[:54] + '%6.2f' % 1.0 + line[60:]
                                chunks.append(line + '\n')
                if cur_start >= 0:
                    f.seek(cur_start)
                    chunks.append(f.read(cur_end - cur_start))
            chunks.append('END\n')
            return ''.join(chunks)

    res_of = spec['res_of']
    run_of = spec['run_of']
    seg = _Segments(path) if run_of is not None else None
    if fmt == 'cif':
        tags = spec['tags']
        chunks = ['data_block\nloop_\n' + ''.join('_atom_site.%s\n' % t for t in tags)]
        rows = cif_rows(spec['path'])
        next(rows)
        iocc = tags.index('occupancy') if 'occupancy' in tags else None
        ib = tags.index('B_iso_or_equiv') if miss_b and 'B_iso_or_equiv' in tags else None
        cols = _columns(spec['path'], tags)
        c_chain, c_comp = cols['chain'], cols['comp']
        has_label = 'label_asym_id' in tags
        ilabel = tags.index('label_asym_id') if has_label else None
        label_is_chain = c_chain == ilabel
        for i, r in enumerate(rows):
            res_i = res_of[i]
            if not in_b[res_i]:
                continue
            if zero_occ and iocc is not None and (r[iocc] in ('.', '?') or float(r[iocc]) == 0):
                r[iocc] = '1.00'
            if ib is not None and r[ib] in ('.', '?'):
                r[ib] = '0.00'
            if seg is not None:
                out = seg.cif_label(int(run_of[res_i]), r[c_chain], r[ilabel] if has_label else r[c_chain],
                                    r[c_comp], label_is_chain, has_label)
                if has_label:
                    r[ilabel] = out
            chunks.append(' '.join(cif_value(t) for t in r) + '\n')
        chunks.append('#\n')
        return ''.join(chunks)
    else:
        chunks = []
        for i, line in enumerate(pdb_lines(spec['path'])):
            res_i = res_of[i]
            if not in_b[res_i]:
                continue
            if zero_occ and not pdb_number(line.ljust(60)[54:60]):
                line = line.ljust(60)[:54] + '%6.2f' % 1.0 + line[60:]
            if seg is not None and seg.pdb_ter(int(run_of[res_i]), line[20:22].strip()):
                chunks.append('TER\n')
            chunks.append(line + '\n')
        chunks.append('END\n')
        return ''.join(chunks)


def write_blocks(model, members, workdir, zero_occupancy_as_one=False, max_open=None, names=None,
                 missing_b_as_zero=False):
    """
    Write the atoms of each set of residues (members[b], residue indices) to
    its own file in workdir, in the model's format and file order; return
    the paths in block order. Files are named names[b] plus the format's
    extension, by default block00000, block00001, ... With
    zero_occupancy_as_one, an occupancy of 0 or none is written as 1; with
    missing_b_as_zero, an mmCIF B of '.' or '?' as 0 (in PDB, iotbx reads a
    blank B as 0 already). Blocks are written in batches of at most max_open
    files, one pass over the atoms per batch, so that a large number of
    blocks stays within the process's open-file limit (by default, soft
    limit minus 64).
    """
    if max_open is None:
        max_open = _nofile_budget()
    member_of = [[] for _ in model.residues]
    for b, residues in enumerate(members):
        for r in residues:
            member_of[int(r)].append(b)
    ext = '.cif' if model.format == 'cif' else '.pdb'
    if names is None:
        names = ['block%05d' % b for b in range(len(members))]
    if len(names) != len(members) or len(set(names)) != len(names):
        raise ValueError('write_blocks needs one distinct name per block')
    paths = [os.path.join(workdir, name + ext) for name in names]
    for start in range(0, len(paths), max_open):
        batch = list(range(start, min(start + max_open, len(paths))))
        _write_batch(model, member_of, paths, batch, zero_occupancy_as_one, missing_b_as_zero)
    return paths
