"""
Pure-Python hierarchy reader and adapter matching iotbx.pdb.hierarchy semantics
(PDB construct_hierarchy, mmCIF pdb_hierarchy_builder, _hd_selection,
merge_atoms_at_end_to_residues, flip_symmetric_amino_acids, and conformers()).
"""
import math
import os
import numpy as np
from .io import (
    InputError,
    PDB_NUMBERS,
    REQUIRED_COLUMNS,
    VALUE_COLUMNS,
    _columns,
    _pdb_records,
    _starts_chain,
    _unreadable,
    cif_rows,
    file_format,
    pdb_number,
)


def _hy36encode_4(value):
    """iotbx.pdb.hy36encode(width=4, value=int(value))"""
    if -999 <= value <= 9999:
        return "%4d" % value
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 10000 <= value < 10000 + 26 * 36 ** 3:
        v = value - 10000 + 10 * 36 ** 3
        out = []
        for _ in range(4):
            out.append(digits[v % 36])
            v //= 36
        return "".join(reversed(out))
    digits_lower = "0123456789abcdefghijklmnopqrstuvwxyz"
    v = value - (10000 + 26 * 36 ** 3) + 10 * 36 ** 3
    out = []
    for _ in range(4):
        out.append(digits_lower[v % 36])
        v //= 36
    return "".join(reversed(out))


def _hy36decode(s):
    """iotbx.pdb.hy36decode for 4-char resseq"""
    st = s.strip()
    if not st:
        return 0
    if st[0] in "-+0123456789":
        try:
            return int(st)
        except ValueError:
            pass
    if s[0].isupper():
        return int(s, 36) - 10 * 36 ** 3 + 10000
    if s[0].islower():
        return int(s, 36) - 10 * 36 ** 3 + 10000 + 26 * 36 ** 3
    return 0


def format_pdb_atom_name(atom_name, atom_type):
    """iotbx.pdb.mmcif.format_pdb_atom_name"""
    if len(atom_name) >= 4:
        return atom_name
    atom_name = atom_name.strip()
    if len(atom_type) == 1:
        if atom_name and atom_name.upper()[0] == atom_type.upper():
            atom_name = " %s" % atom_name
    while len(atom_name) < 4:
        atom_name = "%s " % atom_name
    return atom_name


def _dihedral_deg_single(p0, p1, p2, p3):
    """Exact single-dihedral calculation matching scitbx::math::dihedral_angle."""
    d_01 = p0 - p1
    d_21 = p2 - p1
    d_23 = p2 - p3
    n_0121 = np.cross(d_01, d_21)
    n_2123 = np.cross(d_21, d_23)
    n_0121_norm = np.dot(n_0121, n_0121)
    n_2123_norm = np.dot(n_2123, n_2123)
    denom = np.sqrt(n_0121_norm * n_2123_norm)
    if denom == 0.0:
        return None
    cos_angle = max(-1.0, min(1.0, float(np.dot(n_0121, n_2123) / denom)))
    angle = float(np.arccos(cos_angle) * (180.0 / np.pi))
    if np.dot(d_21, np.cross(n_0121, n_2123)) < 0.0:
        angle = -angle
    return angle


def _dihedral_gt_90_fast(p0, p1, p2, p3):
    """Fast scalar check for abs(dihedral(p0, p1, p2, p3)) > 90 deg matching scitbx."""
    d01x, d01y, d01z = p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]
    d21x, d21y, d21z = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    d23x, d23y, d23z = p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2]
    n1x = d01y * d21z - d01z * d21y
    n1y = d01z * d21x - d01x * d21z
    n1z = d01x * d21y - d01y * d21x
    n2x = d21y * d23z - d21z * d23y
    n2y = d21z * d23x - d21x * d23z
    n2z = d21x * d23y - d21y * d23x
    n1_sq = n1x * n1x + n1y * n1y + n1z * n1z
    n2_sq = n2x * n2x + n2y * n2y + n2z * n2z
    denom = math.sqrt(n1_sq * n2_sq)
    if denom == 0.0:
        return False
    cos_a = max(-1.0, min(1.0, (n1x * n2x + n1y * n2y + n1z * n2z) / denom))
    return math.acos(cos_a) * (180.0 / math.pi) > 90.0


def _chiral_flip_fast(p0, p1, p2, p3):
    """Fast scalar check for chiral volume flip on VAL/LEU matching iotbx."""
    d1x, d1y, d1z = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    d2x, d2y, d2z = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    d3x, d3y, d3z = p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]
    cx = d2y * d3z - d2z * d3y
    cy = d2z * d3x - d2x * d3z
    cz = d2x * d3y - d2y * d3x
    vol = d1x * cx + d1y * cy + d1z * cz
    return abs(-2.5 - vol) > 2.0


class JaxAtom:
    __slots__ = ("name", "element", "xyz", "occ", "b", "i_seq", "hetero", "_parent")

    def __init__(self, name, element, xyz, occ=1.0, b=0.0, i_seq=0, hetero=False, parent=None):
        self.name = name
        self.element = element
        self.xyz = xyz if type(xyz) is tuple and len(xyz) == 3 else (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self.occ = occ if type(occ) is float else float(occ)
        self.b = b if type(b) is float else float(b)
        self.i_seq = i_seq if type(i_seq) is int else int(i_seq)
        self.hetero = bool(hetero)
        self._parent = parent

    def parent(self):
        return self._parent

    def detached_copy(self):
        return JaxAtom(self.name, self.element, self.xyz, self.occ, self.b, self.i_seq, self.hetero)

    def element_is_hydrogen(self):
        return self.element.strip() in ("H", "D")


class JaxAtomGroup:
    __slots__ = ("altloc", "resname", "_atoms", "_parent")

    def __init__(self, altloc="", resname="   ", parent=None):
        self.altloc = altloc
        self.resname = resname
        self._atoms = []
        self._parent = parent

    def parent(self):
        return self._parent

    def atoms(self):
        return self._atoms

    def atoms_size(self):
        return len(self._atoms)

    def append_atom(self, atom):
        atom._parent = self
        self._atoms.append(atom)

    def remove_atom(self, atom):
        if isinstance(atom, int):
            a = self._atoms.pop(atom)
            a._parent = None
        else:
            self._atoms.remove(atom)
            atom._parent = None

    def get_atom(self, name):
        target = name.strip()
        for a in self._atoms:
            if a.name.strip() == target:
                return a
        return None

    def id_str(self):
        chain_id = ""
        resid = ""
        rg = self._parent
        if rg is not None:
            resid = rg.resid()
            ch = rg._parent
            if ch is not None:
                chain_id = ch.id
        return "%1s%3s%2s%5s" % (self.altloc, self.resname, chain_id, resid)

    def detached_copy(self):
        ag = JaxAtomGroup(self.altloc, self.resname)
        for a in self._atoms:
            ag.append_atom(a.detached_copy())
        return ag


class JaxResidueGroup:
    __slots__ = ("resseq", "icode", "link_to_previous", "_atom_groups", "_parent")

    def __init__(self, resseq="   1", icode=" ", link_to_previous=True, parent=None):
        self.resseq = resseq
        self.icode = icode
        self.link_to_previous = link_to_previous
        self._atom_groups = []
        self._parent = parent

    def parent(self):
        return self._parent

    def resid(self):
        return "%4s%1s" % (self.resseq, self.icode)

    def resseq_as_int(self):
        return _hy36decode(self.resseq)

    def atom_groups(self):
        return self._atom_groups

    def atom_groups_size(self):
        return len(self._atom_groups)

    def append_atom_group(self, ag):
        ag._parent = self
        self._atom_groups.append(ag)

    def insert_atom_group(self, idx, ag):
        ag._parent = self
        self._atom_groups.insert(idx, ag)

    def remove_atom_group(self, ag):
        if isinstance(ag, int):
            removed = self._atom_groups.pop(ag)
            removed._parent = None
        else:
            self._atom_groups.remove(ag)
            ag._parent = None

    def atoms(self):
        out = []
        for ag in self._atom_groups:
            out.extend(ag._atoms)
        return out

    def atoms_size(self):
        return sum(len(ag._atoms) for ag in self._atom_groups)

    def have_conformers(self):
        for ag in self._atom_groups:
            if ag.altloc not in ("", " "):
                return True
        return False

    def move_blank_altloc_atom_groups_to_front(self):
        n_blank = 0
        for i in range(len(self._atom_groups)):
            ag = self._atom_groups[i]
            if ag.altloc in ("", " "):
                if i != n_blank:
                    self._atom_groups.pop(i)
                    self._atom_groups.insert(n_blank, ag)
                n_blank += 1
        return n_blank

    def edit_blank_altloc(self):
        n_blank = self.move_blank_altloc_atom_groups_to_front()
        if n_blank == 0:
            return
        blank_name_sets = {}
        for i in range(n_blank):
            ag = self._atom_groups[i]
            ag.altloc = ""
            s = blank_name_sets.setdefault(ag.resname, set())
            for a in ag._atoms:
                s.add(a.name)
        blank_but_alt_sets = {}
        for i in range(n_blank, len(self._atom_groups)):
            ag = self._atom_groups[i]
            bset = blank_name_sets.get(ag.resname)
            if not bset:
                continue
            for a in ag._atoms:
                if a.name in bset:
                    blank_but_alt_sets.setdefault(ag.resname, set()).add(a.name)
        if blank_but_alt_sets:
            n_blank_but_alt = 0
            i_ag = 0
            while i_ag < n_blank:
                ag = self._atom_groups[i_ag]
                bba_set = blank_but_alt_sets.get(ag.resname)
                if not bba_set:
                    i_ag += 1
                    continue
                new_ag = None
                i_atom = 0
                while i_atom < len(ag._atoms):
                    a = ag._atoms[i_atom]
                    if a.name not in bba_set:
                        i_atom += 1
                        continue
                    if new_ag is None:
                        ins_idx = n_blank + n_blank_but_alt
                        new_ag = JaxAtomGroup(" ", ag.resname)
                        self.insert_atom_group(ins_idx, new_ag)
                        n_blank_but_alt += 1
                    ag.remove_atom(i_atom)
                    new_ag.append_atom(a)
                if len(ag._atoms) == 0:
                    self.remove_atom_group(i_ag)
                    n_blank -= 1
                else:
                    i_ag += 1

    def detached_copy(self):
        rg = JaxResidueGroup(self.resseq, self.icode, self.link_to_previous)
        for ag in self._atom_groups:
            rg.append_atom_group(ag.detached_copy())
        return rg

    def conformers(self):
        return _build_conformers_from_residue_groups(self._parent, (self,))


def _build_conformers_from_residue_groups(parent, residue_groups):
    """Exact replica of conformer::build_from_residue_groups (hierarchy.cpp:1988-2097)."""
    altlocs = []
    altloc_indices = {}
    for rg in residue_groups:
        for ag in rg._atom_groups:
            al = ag.altloc
            if al == "":
                continue
            if al not in altloc_indices:
                altloc_indices[al] = len(altlocs)
                altlocs.append(al)
    if not altlocs:
        altlocs.append("")
    n_conf = len(altlocs)
    result = [JaxConformer(al, parent) for al in altlocs]
    for rg in residue_groups:
        resnames_by_conf = [[] for _ in range(n_conf)]
        resname_indices_by_conf = [{} for _ in range(n_conf)]
        atoms_by_conf_resname = [[] for _ in range(n_conf)]
        has_altconf_by_conf_resname = [[] for _ in range(n_conf)]
        for ag in rg._atom_groups:
            al = ag.altloc
            is_main = al == ""
            i_confs = range(n_conf) if is_main else (altloc_indices[al],)
            rn = ag.resname
            for ic in i_confs:
                r_map = resname_indices_by_conf[ic]
                ir = r_map.get(rn)
                if ir is None:
                    ir = r_map[rn] = len(resnames_by_conf[ic])
                    resnames_by_conf[ic].append(rn)
                    atoms_by_conf_resname[ic].append([])
                    has_altconf_by_conf_resname[ic].append(False)
                atoms_by_conf_resname[ic][ir].extend(ag._atoms)
                if not is_main:
                    has_altconf_by_conf_resname[ic][ir] = True
        for ic in range(n_conf):
            for ir, rn in enumerate(resnames_by_conf[ic]):
                res = JaxConformerResidue(
                    resname=rn,
                    resseq=rg.resseq,
                    icode=rg.icode,
                    link_to_previous=rg.link_to_previous,
                    is_pure_main_conf=not has_altconf_by_conf_resname[ic][ir],
                    atoms=atoms_by_conf_resname[ic][ir],
                    parent=result[ic],
                )
                result[ic]._residues.append(res)
    return result


class JaxConformerResidue:
    __slots__ = ("resname", "resseq", "icode", "link_to_previous", "is_pure_main_conf", "_atoms", "_parent")

    def __init__(self, resname, resseq, icode, link_to_previous, is_pure_main_conf, atoms, parent):
        self.resname = resname
        self.resseq = resseq
        self.icode = icode
        self.link_to_previous = link_to_previous
        self.is_pure_main_conf = is_pure_main_conf
        self._atoms = atoms
        self._parent = parent

    def parent(self):
        return self._parent

    def atoms(self):
        return self._atoms

    def atoms_size(self):
        return len(self._atoms)

    def resid(self):
        return "%4s%1s" % (self.resseq, self.icode)

    def resseq_as_int(self):
        return _hy36decode(self.resseq)

    def find_atom_by(self, name):
        target = name.strip()
        for a in self._atoms:
            if a.name.strip() == target:
                return a
        return None


class JaxConformer:
    __slots__ = ("altloc", "_residues", "_parent")

    def __init__(self, altloc, parent):
        self.altloc = altloc
        self._residues = []
        self._parent = parent

    def parent(self):
        return self._parent

    def residues(self):
        return self._residues

    def residues_size(self):
        return len(self._residues)


class JaxChain:
    __slots__ = ("id", "_residue_groups", "_parent")

    def __init__(self, chain_id="", parent=None):
        self.id = chain_id
        self._residue_groups = []
        self._parent = parent

    def parent(self):
        return self._parent

    def residue_groups(self):
        return self._residue_groups

    def residue_groups_size(self):
        return len(self._residue_groups)

    def append_residue_group(self, rg):
        rg._parent = self
        self._residue_groups.append(rg)

    def remove_residue_group(self, rg):
        if isinstance(rg, int):
            removed = self._residue_groups.pop(rg)
            removed._parent = None
        else:
            self._residue_groups.remove(rg)
            rg._parent = None

    def atoms(self):
        out = []
        for rg in self._residue_groups:
            for ag in rg._atom_groups:
                out.extend(ag._atoms)
        return out

    def atoms_size(self):
        return sum(rg.atoms_size() for rg in self._residue_groups)

    def merge_residue_groups(self, primary, secondary):
        altloc_resname_dict = {}
        for ag in primary._atom_groups:
            altloc_resname_dict.setdefault(ag.altloc, {})[ag.resname] = ag
        append_buffer = []
        for i in range(len(secondary._atom_groups) - 1, -1, -1):
            ag = secondary._atom_groups[i]
            primary_ag = altloc_resname_dict.get(ag.altloc, {}).get(ag.resname)
            secondary.remove_atom_group(i)
            if primary_ag is not None:
                for a in list(ag._atoms):
                    ag.remove_atom(a)
                    primary_ag.append_atom(a)
            else:
                append_buffer.append(ag)
        for ag in reversed(append_buffer):
            primary.append_atom_group(ag)
        self.remove_residue_group(secondary)

    def merge_disconnected_residue_groups_with_pure_altloc(self):
        matching = {}
        for i_rg, rg in enumerate(self._residue_groups):
            matching.setdefault((rg.resseq, rg.icode), []).append(i_rg)
        to_merge = []
        for key, i_rgs in matching.items():
            if len(i_rgs) <= 1:
                continue
            altlocs = {"", " "}
            ok = True
            for i_rg in i_rgs:
                rg = self._residue_groups[i_rg]
                for ag in rg._atom_groups:
                    if ag.altloc in altlocs:
                        ok = False
                        break
                    altlocs.add(ag.altloc)
                if not ok:
                    break
            if ok:
                to_merge.append(i_rgs)
        if to_merge:
            rgs_snapshot = list(self._residue_groups)
            for i_rgs in to_merge:
                primary = rgs_snapshot[i_rgs[0]]
                for idx in i_rgs[1:]:
                    self.merge_residue_groups(primary, rgs_snapshot[idx])

    def conformers(self):
        return _build_conformers_from_residue_groups(self, self._residue_groups)

    def detached_copy(self):
        ch = JaxChain(self.id)
        for rg in self._residue_groups:
            ch.append_residue_group(rg.detached_copy())
        return ch


class JaxModel:
    __slots__ = ("id", "_chains", "_parent")

    def __init__(self, model_id="", parent=None):
        self.id = model_id
        self._chains = []
        self._parent = parent

    def parent(self):
        return self._parent

    def chains(self):
        return self._chains

    def append_chain(self, ch):
        ch._parent = self
        self._chains.append(ch)

    def remove_chain(self, ch):
        self._chains.remove(ch)
        ch._parent = None

    def atom_groups(self):
        for ch in self._chains:
            for rg in ch._residue_groups:
                for ag in rg._atom_groups:
                    yield ag

    def detached_copy(self):
        m = JaxModel(self.id)
        for ch in self._chains:
            m.append_chain(ch.detached_copy())
        return m


class JaxHierarchy:
    __slots__ = ("_models", "_flat_index")

    def __init__(self):
        self._models = []
        self._flat_index = None

    def models(self):
        return self._models

    def append_model(self, m):
        m._parent = self
        self._models.append(m)

    def chains(self):
        for m in self._models:
            for ch in m._chains:
                yield ch

    def residue_groups(self):
        for m in self._models:
            for ch in m._chains:
                for rg in ch._residue_groups:
                    yield rg

    def atom_groups(self):
        for m in self._models:
            for ch in m._chains:
                for rg in ch._residue_groups:
                    for ag in rg._atom_groups:
                        yield ag

    def atoms(self):
        out = []
        for m in self._models:
            for ch in m._chains:
                for rg in ch._residue_groups:
                    for ag in rg._atom_groups:
                        out.extend(ag._atoms)
        return out

    def atoms_size(self):
        return sum(ch.atoms_size() for ch in self.chains())

    def reset_i_seq(self):
        self._flat_index = None
        for i, a in enumerate(self.atoms()):
            a.i_seq = i

    def deep_copy(self):
        h = JaxHierarchy()
        for m in self._models:
            h.append_model(m.detached_copy())
        return h

    def remove_hd_in_place(self):
        """Remove H/D atoms in place without duplicating JaxAtom/JaxResidueGroup objects."""
        self._flat_index = None
        i_seq = 0
        for m in self._models:
            for ch in list(m._chains):
                for rg in list(ch._residue_groups):
                    for ag in list(rg._atom_groups):
                        resname_strip = ag.resname.strip()
                        kept = []
                        for a in ag._atoms:
                            e = a.element.strip()
                            if e:
                                is_hd = e in ("H", "D")
                            else:
                                name = a.name.strip().lstrip("0123456789")
                                is_hd = name[:1] in ("H", "D") and resname_strip != name
                            if not is_hd:
                                a.i_seq = i_seq
                                i_seq += 1
                                kept.append(a)
                        ag._atoms = kept
                        if not kept:
                            rg.remove_atom_group(ag)
                    if rg.atom_groups_size() == 0:
                        ch.remove_residue_group(rg)
                if ch.residue_groups_size() == 0:
                    m.remove_chain(ch)
        return self

    def select_non_hd(self):
        """Return a deep copy with H/D atoms removed (matching worker._hd_selection)."""
        h = JaxHierarchy()
        i_seq = 0
        for m in self._models:
            new_m = JaxModel(m.id)
            for ch in m._chains:
                new_ch = JaxChain(ch.id)
                for rg in ch._residue_groups:
                    new_rg = JaxResidueGroup(rg.resseq, rg.icode, rg.link_to_previous)
                    for ag in rg._atom_groups:
                        new_ag = JaxAtomGroup(ag.altloc, ag.resname)
                        resname_strip = ag.resname.strip()
                        for a in ag._atoms:
                            e = a.element.strip()
                            if e:
                                is_hd = e in ("H", "D")
                            else:
                                name = a.name.strip().lstrip("0123456789")
                                is_hd = name[:1] in ("H", "D") and resname_strip != name
                            if not is_hd:
                                ac = a.detached_copy()
                                ac.i_seq = i_seq
                                i_seq += 1
                                new_ag.append_atom(ac)
                        if new_ag.atoms_size() > 0:
                            new_rg.append_atom_group(new_ag)
                    if new_rg.atom_groups_size() > 0:
                        new_ch.append_residue_group(new_rg)
                if new_ch.residue_groups_size() > 0:
                    new_m.append_chain(new_ch)
            h.append_model(new_m)
        return h

    def merge_atoms_at_end_to_residues(self):
        """Exact replica of iotbx.pdb.hierarchy.root.merge_atoms_at_end_to_residues."""
        self._flat_index = None
        for model_id, model in enumerate(self._models):
            residues = {}
            for ag in list(model.atom_groups()):
                key = "%s %s" % (model_id, ag.id_str())
                prev = residues.get(key)
                if prev is not None:
                    for atom in list(ag._atoms):
                        prev.append_atom(atom.detached_copy())
                        ag.remove_atom(atom)
                    rg = ag._parent
                    rg.remove_atom_group(ag)
                    if rg.atom_groups_size() == 0:
                        ch = rg._parent
                        ch.remove_residue_group(rg)
                        if ch.atoms_size() == 0:
                            model.remove_chain(ch)
                else:
                    residues[key] = ag

    def flip_symmetric_amino_acids(self):
        """Exact replica of iotbx.pdb.hierarchy.root.flip_symmetric_amino_acids."""
        self._flat_index = None
        phe_data = (
            "dihedral",
            ("CA", "CB", "CG", "CD1"),
            (("CD1", "CD2"), ("CE1", "CE2"), ("HD1", "HD2"), ("HE1", "HE2"), ("DD1", "DD2"), ("DE1", "DE2")),
        )
        data = {
            "ARG": (
                "dihedral",
                ("CD", "NE", "CZ", "NH1"),
                (("NH1", "NH2"), ("HH11", "HH21"), ("HH12", "HH22"), ("DH11", "DH21"), ("DH12", "DH22")),
            ),
            "ASP": ("dihedral", ("CA", "CB", "CG", "OD1"), (("OD1", "OD2"),)),
            "GLU": ("dihedral", ("CB", "CG", "CD", "OE1"), (("OE1", "OE2"),)),
            "PHE": phe_data,
            "TYR": phe_data,
            "VAL": (
                "chiral",
                ("CB", "CA", "CG1", "CG2"),
                (
                    ("CG1", "CG2"),
                    ("HG11", "HG21"),
                    ("HG12", "HG22"),
                    ("HG13", "HG23"),
                    ("DG11", "DG21"),
                    ("DG12", "DG22"),
                    ("DG13", "DG23"),
                ),
            ),
            "LEU": (
                "chiral",
                ("CG", "CB", "CD1", "CD2"),
                (
                    ("CD1", "CD2"),
                    ("HD11", "HD21"),
                    ("HD12", "HD22"),
                    ("HD13", "HD23"),
                    ("DD11", "DD21"),
                    ("DD12", "DD22"),
                    ("DD13", "DD23"),
                ),
            ),
        }
        for rg in self.residue_groups():
            flip_it = False
            for ag in rg._atom_groups:
                flip_data = data.get(ag.resname)
                if flip_data is None:
                    flip_data = data.get(ag.resname.strip())
                    if flip_data is None:
                        continue
                by_name = {a.name.strip(): a for a in ag._atoms}
                kind, d_names, pairs = flip_data
                if not flip_it:
                    a0 = by_name.get(d_names[0])
                    a1 = by_name.get(d_names[1])
                    a2 = by_name.get(d_names[2])
                    a3 = by_name.get(d_names[3])
                    if a0 is not None and a1 is not None and a2 is not None and a3 is not None:
                        if kind == "dihedral":
                            if _dihedral_gt_90_fast(a0.xyz, a1.xyz, a2.xyz, a3.xyz):
                                flip_it = True
                        else:
                            if _chiral_flip_fast(a0.xyz, a1.xyz, a2.xyz, a3.xyz):
                                flip_it = True
                if flip_it:
                    flips_stored = []
                    for p1_n, p2_n in pairs:
                        atom1 = by_name.get(p1_n)
                        atom2 = by_name.get(p2_n)
                        if atom1 is None and atom2 is None:
                            continue
                        if (atom1 is None) != (atom2 is None):
                            flips_stored = []
                            break
                        flips_stored.append((atom1, atom2))
                    for atom1, atom2 in flips_stored:
                        atom1.xyz, atom2.xyz = atom2.xyz, atom1.xyz
                        atom1.b, atom2.b = atom2.b, atom1.b


def _read_cif_hierarchy(path, check=True, zero_occupancy_as_one=False, missing_b_as_zero=False):
    rows = cif_rows(path)
    tags = next(rows)
    c = _columns(path, tags)
    values = []
    if check:
        for tag in REQUIRED_COLUMNS:
            if tag not in tags:
                raise InputError(
                    "%s: atom_site has no %s, without which iotbx (and so stock MolProbity) "
                    "cannot read the model" % (path, tag)
                )
        values = [
            (tags.index(t), t)
            for t in VALUE_COLUMNS
            if not (
                zero_occupancy_as_one and t == "occupancy"
                or missing_b_as_zero and t == "B_iso_or_equiv"
            )
        ]
    ilabel = tags.index("label_asym_id") if "label_asym_id" in tags else c["chain"]
    ialt = tags.index("label_alt_id") if "label_alt_id" in tags else None
    ib = tags.index("B_iso_or_equiv") if "B_iso_or_equiv" in tags else None
    igroup = tags.index("group_PDB") if "group_PDB" in tags else None
    imodel = c["model"]
    ichain = c["chain"]
    iseq = c["seq"]
    icomp = c["comp"]
    iins = c["ins"]
    iatom = c["atom"]
    ielem = c["element"]
    ix, iy, iz = c["x"], c["y"], c["z"]
    iocc = c["occ"]
    seq_tag = tags[iseq]

    h = JaxHierarchy()
    model = JaxModel("")
    h.append_model(model)

    first_model_num = None
    current_auth = None
    current_label = None
    current_comp = None
    current_seq = None
    current_ins = None
    chain = None
    rg = None
    atom_groups = {}
    i_seq = 0
    seq_cache = {}
    resname_cache = {}
    name_cache = {}
    dot_q = (".", "?")
    ins_dot_q = ("?", ".", None)

    for r in rows:
        if imodel is not None:
            m_val = r[imodel]
            if first_model_num is None:
                first_model_num = m_val
            elif m_val != first_model_num:
                raise InputError("%s has more than one model; split them" % path)
        auth = r[ichain]
        seq = r[iseq]
        if auth in dot_q:
            raise InputError("%s: atom %s has no chain id" % (path, " ".join(r[:4])))
        if seq in dot_q:
            raise InputError("%s: atom %s has no residue number (%s)" % (path, " ".join(r[:4]), seq_tag))
        label = r[ilabel]
        comp = r[icomp]
        ins = r[iins] if iins is not None else " "
        if ins in ins_dot_q:
            ins = " "

        new_chain = current_auth is None or _starts_chain(
            auth, label, current_auth, current_label, current_comp
        )
        if new_chain:
            chain = JaxChain(auth)
            model.append_chain(chain)

        if new_chain or seq != current_seq or ins != current_ins or auth != current_auth:
            resseq = seq_cache.get(seq)
            if resseq is None:
                try:
                    resseq = _hy36encode_4(int(seq))
                except ValueError:
                    resseq = "%4s" % seq
                seq_cache[seq] = resseq
            rg = JaxResidueGroup(resseq=resseq, icode=ins, link_to_previous=True)
            chain.append_residue_group(rg)
            atom_groups = {}

        alt = r[ialt] if ialt is not None else ""
        if alt in dot_q:
            alt = ""
        resname = resname_cache.get(comp)
        if resname is None:
            resname = "%3s" % comp
            resname_cache[comp] = resname
        ag_key = (alt, resname)
        ag = atom_groups.get(ag_key)
        if ag is None:
            ag = JaxAtomGroup(altloc=alt, resname=resname)
            atom_groups[ag_key] = ag
            if alt == "":
                rg.insert_atom_group(0, ag)
            else:
                rg.append_atom_group(ag)

        elem = r[ielem] if ielem is not None else ""
        raw_atom = r[iatom]
        ne_pair = name_cache.get((raw_atom, elem))
        if ne_pair is None:
            ne_pair = (format_pdb_atom_name(raw_atom, elem), "%2s" % elem)
            name_cache[(raw_atom, elem)] = ne_pair
        atom_name, elem_padded = ne_pair

        try:
            x, y, z = float(r[ix]), float(r[iy]), float(r[iz])
            occ_str = r[iocc] if iocc is not None else "1.0"
            occ = float(occ_str)
            b_str = r[ib] if ib is not None else "0.0"
            b = float(b_str)
        except ValueError:
            if check:
                for idx, _ in values:
                    if r[idx] in dot_q:
                        raise _unreadable(
                            path,
                            "%s (residue %s %s %s)" % (" ".join(r[:4]), auth, comp, seq),
                            [(tag, r[j]) for j, tag in values if r[j] in dot_q],
                        )
            x, y, z = float(r[ix]), float(r[iy]), float(r[iz])
            occ_str = r[iocc] if iocc is not None else "1.0"
            occ = 1.0 if occ_str in dot_q else float(occ_str)
            b_str = r[ib] if ib is not None else "0.0"
            b = 0.0 if b_str in dot_q else float(b_str)

        if zero_occupancy_as_one and occ == 0.0:
            occ = 1.0
        hetero = (igroup is not None) and (r[igroup] == "HETATM")
        ag.append_atom(JaxAtom(atom_name, elem_padded, (x, y, z), occ, b, i_seq, hetero))
        i_seq += 1

        current_auth = auth
        current_label = label
        current_comp = comp
        current_seq = seq
        current_ins = ins

    if i_seq == 0:
        raise InputError("%s has no atoms" % path)
    return h


def _append_pdb_rg(chain_atoms, rg_start, rg_end, chain, link_to_previous, altloc_resname_indices, i_seq_start):
    first = chain_atoms[rg_start]
    rg = JaxResidueGroup(resseq=first["resseq"], icode=first["icode"], link_to_previous=link_to_previous)
    chain.append_residue_group(rg)
    items = sorted(altloc_resname_indices.items(), key=lambda kv: kv[1][0])
    i_seq = i_seq_start
    for (alt, resname), indices in items:
        ag = JaxAtomGroup(altloc=alt, resname=resname)
        rg.append_atom_group(ag)
        for idx in indices:
            a = chain_atoms[rg_start + idx]["atom"]
            a.i_seq = i_seq
            i_seq += 1
            ag.append_atom(a)
    altloc_resname_indices.clear()
    rg.edit_blank_altloc()
    return i_seq


def _read_pdb_hierarchy(path, check=True, zero_occupancy_as_one=False):
    h = JaxHierarchy()
    model = JaxModel("")
    h.append_model(model)

    chains_raw = []
    cur_chain_atoms = None
    cur_chain_id = None
    ended = False

    for line in _pdb_records(path):
        if line.startswith("TER"):
            ended = True
            continue
        line = line.ljust(80)
        cid = line[20:22].strip()
        if cur_chain_id is None or ended or cid != cur_chain_id:
            cur_chain_atoms = []
            chains_raw.append((cid, cur_chain_atoms))
            cur_chain_id = cid
            ended = False
        values = {}
        for column, i, j in PDB_NUMBERS:
            v = values[column] = pdb_number(line[i:j])
            if v is None and (check or column in ("x", "y", "z")):
                raise InputError(
                    "%s: atom %s has '%s' in columns %d-%d (%s), which iotbx (and so stock "
                    "MolProbity) cannot read"
                    % (path, line[:27].strip(), line[i:j].strip(), i + 1, j, column)
                )
        occ = values["occupancy"] if values["occupancy"] is not None else 1.0
        if zero_occupancy_as_one and occ == 0.0:
            occ = 1.0
        b = values["B"] if values["B"] is not None else 0.0
        name = line[12:16]
        alt = line[16]
        resname = line[17:20]
        resseq = line[22:26]
        icode = line[26]
        elem = line[76:78]
        hetero = line.startswith("HETATM")
        atom = JaxAtom(name, elem, (values["x"], values["y"], values["z"]), occ, b, 0, hetero)
        cur_chain_atoms.append(
            {
                "atom": atom,
                "altloc": alt,
                "resname": resname,
                "resseq": resseq,
                "icode": icode,
                "resid": resseq + icode,
            }
        )

    i_seq_start = 0
    for cid, catoms in chains_raw:
        if not catoms:
            continue
        chain = JaxChain(cid)
        model.append_chain(chain)
        altloc_resname_indices = {}
        rg_start = 0
        link_to_previous = False
        prev_resid = None
        prev_resname = None
        open_resname_run_has_blank_altloc = False
        n_c = len(catoms)
        for i_atom in range(n_c):
            ial = catoms[i_atom]
            resid = ial["resid"]
            resname = ial["resname"]
            curr_blank = ial["altloc"] == " "
            if prev_resid is not None:
                is_boundary = prev_resid != resid
                if not is_boundary and prev_resname != resname:
                    if open_resname_run_has_blank_altloc or curr_blank:
                        is_boundary = True
                    else:
                        for j_atom in range(i_atom + 1, n_c):
                            fwd = catoms[j_atom]
                            if fwd["resname"] != resname or fwd["resid"] != resid:
                                break
                            if fwd["altloc"] == " ":
                                is_boundary = True
                                break
                if is_boundary:
                    i_seq_start = _append_pdb_rg(
                        catoms, rg_start, i_atom, chain, link_to_previous, altloc_resname_indices, i_seq_start
                    )
                    rg_start = i_atom
                    link_to_previous = True
                    open_resname_run_has_blank_altloc = False
            prev_resid = resid
            prev_resname = resname
            if curr_blank:
                open_resname_run_has_blank_altloc = True
            altloc_resname_indices.setdefault((ial["altloc"], resname), []).append(i_atom - rg_start)
        if prev_resid is not None:
            i_seq_start = _append_pdb_rg(
                catoms, rg_start, n_c, chain, link_to_previous, altloc_resname_indices, i_seq_start
            )
        chain.merge_disconnected_residue_groups_with_pure_altloc()

    if i_seq_start == 0:
        raise InputError("%s has no atoms" % path)
    return h


def read_hierarchy(path, check=True, zero_occupancy_as_one=False, missing_b_as_zero=False):
    """Read a PDB or mmCIF file into a JaxHierarchy matching iotbx.pdb.input.construct_hierarchy()."""
    path = os.fspath(path)
    if file_format(path) == "cif":
        return _read_cif_hierarchy(path, check, zero_occupancy_as_one, missing_b_as_zero)
    return _read_pdb_hierarchy(path, check, zero_occupancy_as_one)


def prepare_noh_hierarchy(hierarchy, in_place=False):
    """
    Prepare a non-H hierarchy matching worker.py lines 87-91:
    select(~hd) -> merge_atoms_at_end_to_residues() -> reset_i_seq() -> flip_symmetric_amino_acids().
    """
    noh = hierarchy.remove_hd_in_place() if in_place else hierarchy.select_non_hd()
    noh.merge_atoms_at_end_to_residues()
    noh.reset_i_seq()
    noh.flip_symmetric_amino_acids()
    return noh
