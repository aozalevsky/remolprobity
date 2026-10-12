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


def cif_value(tok):
    """tok written as a CIF value"""
    if tok and not any(c.isspace() for c in tok) and tok[0] not in '\'"_#$;[]' \
            and tok.lower() not in ('loop_', 'stop_', 'global_') \
            and not tok.lower().startswith(('data_', 'save_')):
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


def cif_rows(path):
    """The atom_site loop of an mmCIF file: its tags, then its rows as token lists"""
    tags = []
    with _open(path) as f:
        it = iter(f)
        line = ''
        for line in it:
            if line.startswith('_atom_site.'):
                tags.append(line.split()[0][len('_atom_site.'):])
            elif tags:
                break
        if not tags:
            raise InputError('%s has no atom_site loop' % path)
        yield tags
        row = []
        while line is not None:
            if line.startswith(';'):
                raise InputError('%s: text fields in atom_site are not supported' % path)
            if line.startswith(('#', 'loop_', '_', 'data_')):
                break
            row += cif_tokens(line)
            while len(row) >= len(tags):
                yield row[:len(tags)]
                row = row[len(tags):]
            line = next(it, None)
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

    @property
    def n_atoms(self):
        return len(self.xyz)

    def atoms_per_residue(self):
        return np.bincount(self.res_of, minlength=len(self.residues))


class _Table:
    """Atoms as they are read, into numpy chunks"""

    def __init__(self):
        self.xyz_chunks, self.res_chunks, self.xyz, self.res = [], [], [], []
        self.index, self.residues, self.resnames, self.ca = {}, [], [], {}
        self.chains, self.names, self.runs = {}, {}, []
        self.occupancy_max, self.zero_occupancy, self.hydrogens = 0.0, 0, 0

    def add(self, chain, resseq, icode, resname, name, element, xyz, occupancy, run=0):
        key = (chain, residue_id(resseq, icode))
        k = self.index.get(key)
        if k is None:
            k = self.index[key] = len(self.residues)
            self.residues.append(key)
            self.runs.append(run)  # a residue is in the run of its first atom
            self.resnames.append(resname.strip())
        self.res.append(k)
        self.xyz.append(xyz)
        if name == 'CA' and k not in self.ca:
            self.ca[k] = xyz
        self.chains.setdefault(chain, None)
        self.names.setdefault(resname, None)
        if not occupancy:
            self.zero_occupancy += 1
        elif occupancy > self.occupancy_max:
            self.occupancy_max = occupancy
        if element in ('H', 'D'):
            self.hydrogens += 1
        if len(self.res) >= CHUNK:
            self.flush()

    def flush(self):
        if self.res:
            self.xyz_chunks.append(np.array(self.xyz, dtype=np.float64))
            self.res_chunks.append(np.array(self.res, dtype=np.int64))
            self.xyz, self.res = [], []

    def model(self, path, fmt, tags):
        self.flush()
        if not self.residues:
            raise InputError('%s has no atoms' % path)
        xyz = np.concatenate(self.xyz_chunks)
        res_of = np.concatenate(self.res_chunks)
        counts = np.bincount(res_of, minlength=len(self.residues))
        position = np.zeros((len(self.residues), 3))
        np.add.at(position, res_of, xyz)
        position /= counts[:, None]
        for k, p in self.ca.items():
            position[k] = p
        return Model(path=path, format=fmt, sha256=sha256(path), tags=tags, xyz=xyz,
                     res_of=res_of, residues=self.residues, resnames=self.resnames,
                     position=position, chain_ids=list(self.chains),
                     residue_names=list(self.names), occupancy_max=self.occupancy_max,
                     zero_occupancy=self.zero_occupancy, hydrogens=self.hydrogens,
                     run_of=np.array(self.runs, dtype=np.int64))


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
         'occ': pick('occupancy'), 'model': pick('pdbx_PDB_model_num')}
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
    rows = cif_rows(path)
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
    t = _Table()
    model = None
    run, prev = -1, None
    ilabel = tags.index('label_asym_id') if 'label_asym_id' in tags else c['chain']
    for r in rows:
        if c['model'] is not None:
            if model is None:
                model = r[c['model']]
            elif r[c['model']] != model:
                raise InputError('%s has more than one model; split them' % path)
        chain, seq = r[c['chain']], r[c['seq']]
        if chain in ('.', '?'):
            raise InputError('%s: atom %s has no chain id' % (path, ' '.join(r[:4])))
        if seq in ('.', '?'):
            raise InputError('%s: atom %s has no residue number (%s)'
                             % (path, ' '.join(r[:4]), tags[c['seq']]))
        if check:
            for i, _ in values:
                if r[i] in ('.', '?'):
                    raise _unreadable(path, '%s (residue %s %s %s)' % (' '.join(r[:4]), chain, r[c['comp']], seq),
                                      [(tag, r[j]) for j, tag in values if r[j] in ('.', '?')])
        ins = r[c['ins']] if c['ins'] is not None else '?'
        occ = r[c['occ']] if c['occ'] is not None else '?'
        if prev is None or _starts_chain(chain, r[ilabel], prev[0], prev[1], prev[2]):
            run += 1
        prev = (chain, r[ilabel], r[c['comp']])
        t.add(chain, seq, '' if ins in ('.', '?') else ins, '%3s' % r[c['comp']],
              r[c['atom']], r[c['element']].upper() if c['element'] is not None else '',
              (float(r[c['x']]), float(r[c['y']]), float(r[c['z']])),
              None if occ in ('.', '?') else float(occ), run)
    return t.model(path, 'cif', tags)


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


def _pdb_records(path):
    """ATOM, HETATM and TER lines of a PDB file; refuses a second MODEL"""
    models = 0
    with _open(path) as f:
        for line in f:
            if line.startswith('MODEL'):
                models += 1
                if models > 1:
                    raise InputError('%s has more than one model; split them' % path)
            elif line.startswith(('ATOM  ', 'HETATM', 'TER')):
                yield line.rstrip('\n')


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
    t = _Table()
    run, chain, ended = 0, None, False
    for line in _pdb_records(path):
        if line.startswith('TER'):
            ended = True  # iotbx starts a chain object after a TER
            continue
        line = line.ljust(80)
        if chain is not None and (ended or line[20:22].strip() != chain):
            run += 1
        chain, ended = line[20:22].strip(), False
        values = {}
        for column, i, j in PDB_NUMBERS:
            v = values[column] = pdb_number(line[i:j])
            if v is None and (check or column in ('x', 'y', 'z')):
                raise InputError("%s: atom %s has '%s' in columns %d-%d (%s), which iotbx (and so stock "
                                 "MolProbity) cannot read" % (path, line[:27].strip(), line[i:j].strip(),
                                                                 i + 1, j, column))
        name = line[12:16].strip()
        element = line[76:78].strip().upper() or name[:1].upper()
        t.add(line[20:22].strip(), line[22:26], line[26], line[17:20], name, element,
              (values['x'], values['y'], values['z']), values['occupancy'], run)
    return t.model(path, 'pdb', [])


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
            ilabel = model.tags.index('label_asym_id') if 'label_asym_id' in model.tags else None
            label_is_chain = cols['chain'] == ilabel
            segments = [_Segments(paths[b]) for b in batch] if model.run_of is not None else None
            for i, r in enumerate(rows):
                if zero_occupancy_as_one and iocc is not None and \
                        (r[iocc] in ('.', '?') or float(r[iocc]) == 0):
                    r[iocc] = '1.00'
                if ib is not None and r[ib] in ('.', '?'):
                    r[ib] = '0.00'
                line = None if segments is not None else ' '.join(cif_value(t) for t in r) + '\n'
                for b in member_of[res_of[i]]:
                    if b in local:
                        if segments is not None:
                            row = list(r)
                            if ilabel is not None:
                                row[ilabel] = segments[local[b]].cif_label(
                                    int(model.run_of[res_of[i]]), r[cols['chain']], r[ilabel],
                                    r[cols['comp']], label_is_chain, True)
                            else:
                                segments[local[b]].cif_label(
                                    int(model.run_of[res_of[i]]), r[cols['chain']], r[cols['chain']],
                                    r[cols['comp']], True, False)
                            files[local[b]].write(' '.join(cif_value(t) for t in row) + '\n')
                        else:
                            files[local[b]].write(line)
            for f in files:
                f.write('#\n')
        else:
            segments = [_Segments(paths[b]) for b in batch] if model.run_of is not None else None
            for i, line in enumerate(pdb_lines(model.path)):
                if zero_occupancy_as_one:
                    if not pdb_number(line.ljust(60)[54:60]):  # blank or 0, as iotbx reads it
                        line = line.ljust(60)[:54] + '%6.2f' % 1.0 + line[60:]
                line += '\n'
                for b in member_of[res_of[i]]:
                    if b in local:
                        if segments is not None and segments[local[b]].pdb_ter(
                                int(model.run_of[res_of[i]]), line[20:22].strip()):
                            files[local[b]].write('TER\n')
                        files[local[b]].write(line)
            for f in files:
                f.write('END\n')
    finally:
        for f in files:
            f.close()


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
