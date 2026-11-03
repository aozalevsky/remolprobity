"""
GPU-accelerated all-atom clash detection (probe) in JAX.

Reproduces MolProbity's `probe -u -q -mc -het -once -NOVDWOUT -CON "ogt10 not water" "ogt10"`
and `-dumpatominfo "ogt10 not water"` on GPU using static tables exported from probe 2.26
(`~/.cache/remolprobity/jax_probe.npz`).
"""
import functools
import json
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .clashes import (
    REACH,
    MarginError,
    _atoms_by_residue,
    _cliques,
    _coordinates,
    check_cliques,
    group_label,
    moved_points,
    rename_chain_ids,
    restore_chain_ids,
    spare_chain_ids,
)
from .ids import residue_id
from .jax_tables import (
    _get_jax,
    build_probe_npz,
    call_gpu,
    default_probe_path,
    lazy_jit,
    register_gpu_op,
)
from .naming import pdb_copy
from .tools import add_hydrogens

DEFAULT_PROBE_NPZ_PATH = default_probe_path()

PROBE_RAD = np.float32(0.25)
COVRADFUDGE = np.float32(0.2)
OCCUPANCY_CUTOFF = np.float32(0.02)
LOW_GOOD_CUT = np.float32(-0.4)
MIN_REGULAR_HB_CUTOFF = np.float32(0.6)
MIN_CHARGED_HB_CUTOFF = np.float32(0.8)


@dataclass
class ProbeClashAtom:
    chain_id: str
    resseq: str
    icode: str
    resname: str
    name: str
    altloc: str
    model_id: str = ""

    def id_str(self):
        return f"{self.chain_id:2s}{self.resseq:4s}{self.icode:1s}{self.resname:3s} {self.name:4s}{self.altloc:1s}"


@dataclass
class ProbeClash:
    atoms_info: list
    overlap: float
    probe_type: str = "bo"
    max_b_factor: float = 0.0

    def id_str(self, spacer=" "):
        return f"{self.atoms_info[0].id_str()}{spacer}{self.atoms_info[1].id_str()}"

    def __lt__(self, other):
        return self.overlap < other.overlap


def decode_atom_string(atom_str, use_segids=False, model_id=""):
    """Match mmtbx.validation.clashscore.decode_atom_string."""
    if not use_segids or len(atom_str) == 16:
        return ProbeClashAtom(
            chain_id=atom_str[0:2],
            resseq=atom_str[2:6],
            icode=atom_str[6],
            resname=atom_str[7:10],
            name=atom_str[11:15],
            altloc=atom_str[15],
            model_id=model_id,
        )
    return ProbeClashAtom(
        chain_id=atom_str[0:4],
        resseq=atom_str[4:8],
        icode=atom_str[8],
        resname=atom_str[9:12],
        name=atom_str[13:17],
        altloc=atom_str[17],
        model_id=model_id,
    )


def _v3scale_array(vecs, newlen):
    """Match geom3d.c:v3scale on an array of 3D vectors (N, 3)."""
    lengths = np.linalg.norm(vecs, axis=-1, keepdims=True)
    scale = np.where(lengths != 0.0, newlen / np.where(lengths != 0.0, lengths, 1.0), 0.0)
    return vecs * scale


@functools.lru_cache(maxsize=1)
def load_cpu_probe_tables(npz_path=None):
    """Load static probe tables and precomputed dot spheres as NumPy arrays (no JAX import)."""
    path = npz_path or default_probe_path()
    if not os.path.isfile(path):
        build_probe_npz(path, only_if_missing=True)
    npz = np.load(path, allow_pickle=False)
    meta = json.loads(str(npz["meta_json"]))
    atom_id = meta["atom_id_map"]
    props_map = meta["props_map"]
    stdconn = meta["stdconn"]

    atno_arr = npz["atno"]
    erad_arr = npz["erad"]
    covrad_arr = npz["covrad"]
    elem_flags_arr = npz["elem_flags"]
    num_types = len(atno_arr)

    # Map distinct heavy target atom names in stdconn to bit indices 0..66
    heavy_targets = sorted(
        {
            part
            for k, v in stdconn.items()
            if ("H" in k.split(":")[1] or "D" in k.split(":")[1])
            for part in v.split(":")
            if len(part) == 4
        }
    )
    heavy_bit_index = {name: idx for idx, name in enumerate(heavy_targets)}
    other_heavy_bit = len(heavy_targets)  # bit 66 for any heavy atom name not in stdconn targets

    def make_bit_pair(bit_positions):
        w0 = np.uint64(0)
        w1 = np.uint64(0)
        for b in bit_positions:
            if b < 64:
                w0 |= np.uint64(1) << np.uint64(b)
            else:
                w1 |= np.uint64(1) << np.uint64(b - 64)
        return (w0, w1)

    heavy_name_to_bits = {
        name: make_bit_pair([idx]) for name, idx in heavy_bit_index.items()
    }
    default_heavy_bits = make_bit_pair([other_heavy_bit])
    all_heavy_bits = (np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64(0xFFFFFFFFFFFFFFFF))

    bondedto_str_to_bits = {}
    for v in set(stdconn.values()):
        bits = [heavy_bit_index[p] for p in v.split(":") if p in heavy_bit_index]
        bondedto_str_to_bits[v] = make_bit_pair(bits)

    # Precompute dot spheres and expanded probe-center vectors (expvect)
    # Index 0..num_types-1 are AtomTbl types; index num_types is COdots (carbonyl C)
    co_rad = float(np.float32(1.70) * np.float32(1.65 / 1.70))
    dot_vecs_list = []
    exp_vecs_list = []
    dot_counts = np.zeros(num_types + 1, dtype=np.int32)
    for i in range(num_types):
        d = npz[f"dots_{i}"].astype(np.float64)
        r = float(erad_arr[i])
        e = _v3scale_array(d, float(np.float32(r) + PROBE_RAD)) if len(d) else d
        dot_vecs_list.append(d)
        exp_vecs_list.append(e)
        dot_counts[i] = len(d)
    d_co = npz["dots_co"].astype(np.float64)
    e_co = _v3scale_array(d_co, float(np.float32(co_rad) + PROBE_RAD))
    dot_vecs_list.append(d_co)
    exp_vecs_list.append(e_co)
    dot_counts[num_types] = len(d_co)

    # Build padded arrays for 3 dot-count tiers:
    # Tier 0: D <= 320 (all H atoms, metals, ions)
    # Tier 1: D <= 656 (C, N, O, S, P, Se, Cl, Br)
    # Tier 2: D <= 1216 (I, Xe, Rn, etc.)
    tiers = {}
    for max_d in (320, 656, 1216):
        dv = np.zeros((num_types + 1, max_d, 3), dtype=np.float64)
        ev = np.zeros((num_types + 1, max_d, 3), dtype=np.float64)
        dm = np.zeros((num_types + 1, max_d), dtype=bool)
        for i in range(num_types + 1):
            n = min(len(dot_vecs_list[i]), max_d)
            if n > 0:
                dv[i, :n] = dot_vecs_list[i][:n]
                ev[i, :n] = exp_vecs_list[i][:n]
                dm[i, :n] = True
        tiers[max_d] = (dv, ev, dm)

    return {
        "meta": meta,
        "atom_id": atom_id,
        "props_map": props_map,
        "res_macros": meta["res_macros"],
        "stdconn": stdconn,
        "atno": atno_arr,
        "erad": erad_arr,
        "covrad": covrad_arr,
        "elem_flags": elem_flags_arr,
        "co_rad": co_rad,
        "co_dot_idx": num_types,
        "dot_counts": dot_counts,
        "tiers": tiers,
        "heavy_name_to_bits": heavy_name_to_bits,
        "default_heavy_bits": default_heavy_bits,
        "all_heavy_bits": all_heavy_bits,
        "bondedto_str_to_bits": bondedto_str_to_bits,
    }


@functools.lru_cache(maxsize=1)
def load_probe_tables(npz_path=None):
    """Load static probe tables with device arrays for the 3 dot-count tiers."""
    _, jnp = _get_jax()
    cpu_tbl = load_cpu_probe_tables(npz_path)
    out = dict(cpu_tbl)
    out["tiers"] = {
        max_d: (jnp.asarray(dv), jnp.asarray(ev), jnp.asarray(dm))
        for max_d, (dv, ev, dm) in cpu_tbl["tiers"].items()
    }
    return out


def hy36decode4(s):
    """Decode a 4-character PDB residue number (decimal or hybrid-36)."""
    s_strip = s.strip()
    if not s_strip:
        return 0
    if s_strip[0] in "-0123456789":
        try:
            return int(s_strip)
        except ValueError:
            pass
    c0 = s[0]
    if "A" <= c0 <= "Z":
        val = 0
        for ch in s:
            d = ord(ch) - ord("0") if "0" <= ch <= "9" else ord(ch.upper()) - ord("A") + 10
            val = val * 36 + d
        return val - 10 * (36**3) + 10000
    if "a" <= c0 <= "z":
        val = 0
        for ch in s:
            d = ord(ch) - ord("0") if "0" <= ch <= "9" else ord(ch.lower()) - ord("a") + 10
            val = val * 36 + d
        return val + 16 * (36**3) + 10000
    return 0


def load_probe_atoms(pdb_path, ogt=10):
    """
    Load atoms from a PDB file and run probe's exact property assignment,
    chain-end/histidine charge checks, selection flags, and updateHydrogenInfo.
    """
    tbl = load_cpu_probe_tables()
    atom_id = tbl["atom_id"]
    props_map = tbl["props_map"]
    res_macros = tbl["res_macros"]
    stdconn = tbl["stdconn"]
    atno_arr = tbl["atno"]
    erad_arr = tbl["erad"]
    covrad_arr = tbl["covrad"]
    elem_flags_arr = tbl["elem_flags"]
    meta = tbl["meta"]
    co_rad = tbl["co_rad"]

    ignoreAtom = atom_id["ignoreAtom"]
    atomH = atom_id["atomH"]
    atomHarom = atom_id["atomHarom"]
    atomHpolar = atom_id["atomHpolar"]
    atomHOd = atom_id["atomHOd"]
    atomC = atom_id["atomC"]
    atomCarom = atom_id["atomCarom"]
    atomN = atom_id["atomN"]
    atomO = atom_id["atomO"]
    atomF = atom_id["atomF"]
    atomS = atom_id["atomS"]
    atomP = atom_id["atomP"]
    atomCl = atom_id["atomCl"]
    atomBr = atom_id["atomBr"]
    atomI = atom_id["atomI"]
    atomHe = atom_id["atomHe"]
    atomHf = atom_id["atomHf"]
    atomHg = atom_id["atomHg"]
    atomHo = atom_id["atomHo"]
    atomSe = atom_id["atomSe"]

    ALPHA_PROP = props_map["ALPHA_PROP"]
    BETA_PROP = props_map["BETA_PROP"]
    MC_PROP = props_map["MC_PROP"]
    SC_PROP = props_map["SC_PROP"]
    N_PROP = props_map["N_PROP"]
    C_PROP = props_map["C_PROP"]
    O_PROP = props_map["O_PROP"]
    S_PROP = props_map["S_PROP"]
    P_PROP = props_map["P_PROP"]
    H_PROP = props_map["H_PROP"]
    WATER_PROP = props_map["WATER_PROP"]
    HET_PROP = props_map["HET_PROP"]
    PROT_PROP = props_map["PROT_PROP"]
    DNA_PROP = props_map["DNA_PROP"]
    METAL_PROP = props_map["METAL_PROP"]
    ION_PROP = props_map["ION_PROP"]
    METHYL_PROP = props_map["METHYL_PROP"]
    METHYLENE_PROP = props_map["METHYLENE_PROP"]
    DONOR_PROP = props_map["DONOR_PROP"]
    ACCEPTOR_PROP = props_map["ACCEPTOR_PROP"]
    AROMATIC_PROP = props_map["AROMATIC_PROP"]
    CH_DONOR_PROP = props_map["CH_DONOR_PROP"]
    TEST_ACCEPT_ANGLE_PROP = props_map["TEST_ACCEPT_ANGLE_PROP"]
    RHPHOBIC_PROP = props_map["RHPHOBIC_PROP"]
    RHPHILIC_PROP = props_map["RHPHILIC_PROP"]
    RCHARGED_PROP = props_map["RCHARGED_PROP"]
    POSITIVE_PROP = props_map["POSITIVE_PROP"]
    NEGATIVE_PROP = props_map["NEGATIVE_PROP"]
    MAYBECHG_PROP = props_map["MAYBECHG_PROP"]
    CHECK_ENDS_PROP = props_map["CHECK_ENDS_PROP"]
    AMBIGWATER_PROP = props_map["AMBIGWATER_PROP"]

    AAList = meta["AAList"]
    HphobicAAList = meta["HphobicAAList"]
    HphilicAAList = meta["HphilicAAList"]
    AromaticAAList = meta["AromaticAAList"]
    NAList = meta["NAList"]
    NAbackboneList = meta["NAbackboneList"]
    MethylResList = meta["MethylResList"]
    MethyleneResList = meta["MethyleneResList"]
    ChargedAAList = meta["ChargedAAList"]
    AlwaysChargedAtomsList = meta["AlwaysChargedAtomsList"]
    AmbigChargedAtomsList = meta["AmbigChargedAtomsList"]
    ChargedNucAcidAtomsList = meta["ChargedNucAcidAtomsList"]
    WaterList = meta["WaterList"]
    AromaticAtomsTbl = meta["AromaticAtomsTbl"]
    MethylAtomsTbl = meta["MethylAtomsTbl"]
    MethyleneAtomsTbl = meta["MethyleneAtomsTbl"]
    ChargedAAAtomsTbl = meta["ChargedAAAtomsTbl"]
    DonorAcceptorAtomTbl = meta["DonorAcceptorAtomTbl"]

    na_alt_list = (
        ":GUA:GTP:GDP:GMP:GSP:ADE:ATP:ADP:AMP:CYT:CTP:CDP:CMP:"
        "URA:UTP:UDP:UMP:THY:TTP:TDP:TMP: DA: DT: DC: DG: AR: UR: TR: CR: GR:"
    )

    def fix_atom_name(atomname, resname, pos):
        resn = f":{resname:<3.3s}:"
        ch = atomname[pos].upper() if pos < len(atomname) else " "
        order = ["E", "F", "G", "O", "S"]
        macro_keys = ["HE_RESNAMES", "HF_RESNAMES", "HG_RESNAMES", "HO_RESNAMES", "HS_RESNAMES"]
        if ch in order:
            for k in macro_keys[order.index(ch) :]:
                if resn in res_macros[k]:
                    return True
        return False

    @functools.lru_cache(maxsize=4096)
    def identify_atom(name, resname):
        c0, c1, c2, c3 = name[0], name[1], name[2], name[3]
        n = -1
        if c0 in "*'\"`_+- 0123456789":
            if c1 == "A":
                if c3 == "1":
                    n = atomO
                elif c3 == "2":
                    n = atomN
            elif c1 == "B":
                n = atom_id["atomB"]
            elif c1 == "C":
                n = atomC
            elif c1 == "D":
                n = atomH
            elif c1 == "F":
                n = atomF
            elif c1 == "H":
                if c2 == "E":
                    n = atomHe if fix_atom_name(name, resname, 2) else atomH
                elif c2 == "F":
                    n = atomHf if fix_atom_name(name, resname, 2) else atomH
                elif c2 == "G":
                    n = atomHg if fix_atom_name(name, resname, 2) else atomH
                elif c2 == "O":
                    n = atomHo if fix_atom_name(name, resname, 2) else atomH
                else:
                    n = atomH
            elif c1 == "I":
                n = atomI
            elif c1 == "K":
                n = atom_id["atomK"]
            elif c1 == "N":
                n = atomN
            elif c1 == "O":
                n = atomO
            elif c1 == "P":
                n = atomP
            elif c1 == "S":
                if c0 == " " and c2 == "E":
                    n = atomSe
                else:
                    n = atomS
            elif c1 == "U":
                n = atom_id["atomU"]
            elif c1 == "V":
                n = atom_id["atomV"]
            elif c1 == "W":
                n = atom_id["atomW"]
            elif c1 == "Y":
                n = atom_id["atomY"]
        elif c0 == "A":
            n = {
                "C": atomC,
                "G": atom_id["atomAg"],
                "H": atomH,
                "L": atom_id["atomAl"],
                "M": atom_id["atomAm"],
                "N": atomN,
                "O": atomO,
                "P": atomP,
                "R": atom_id["atomAr"],
                "S": atom_id["atomAs"],
                "T": atom_id["atomAt"],
                "U": atom_id["atomAu"],
            }.get(c1, -1)
        elif c0 == "B":
            n = {
                "A": atom_id["atomBa"],
                "E": atom_id["atomBe"],
                "I": atom_id["atomBi"],
                "K": atom_id["atomBk"],
                "R": atomBr,
            }.get(c1, -1)
        elif c0 == "C":
            n = {
                "A": atom_id["atomCa"],
                "C": atomC,
                "D": atom_id["atomCd"],
                "E": atom_id["atomCe"],
                "F": atom_id["atomCf"],
                "H": atomH,
                "L": atomCl,
                "M": atom_id["atomCm"],
                "N": atomN,
                "O": atom_id["atomCo"],
                "P": atomP,
                "R": atom_id["atomCr"],
                "S": atom_id["atomCs"],
                "U": atom_id["atomCu"],
            }.get(c1, atomC)
        elif c0 == "D":
            n = {"Y": atom_id["atomDy"], "C": atomC, "H": atomH, "N": atomN, "O": atomO, "P": atomP}.get(c1, atomH)
        elif c0 == "E":
            n = {
                "R": atom_id["atomEr"],
                "S": atom_id["atomEs"],
                "U": atom_id["atomEu"],
                "C": atomC,
                "H": atomH,
                "N": atomN,
                "O": atomO,
                "P": atomP,
            }.get(c1, -1)
        elif c0 == "F":
            n = {
                "E": atom_id["atomFe"],
                "M": atom_id["atomFm"],
                "R": atom_id["atomFr"],
                "C": atomC,
                "H": atomH,
                "N": atomN,
                "O": atomO,
                "P": atomP,
            }.get(c1, -1)
        elif c0 == "G":
            n = {
                "A": atom_id["atomGa"],
                "D": atom_id["atomGd"],
                "E": atom_id["atomGe"],
                "C": atomC,
                "H": atomH,
                "N": atomN,
                "O": atomO,
                "P": atomP,
            }.get(c1, -1)
        elif c0 == "H":
            if c1 == "E":
                n = atomHe if fix_atom_name(name, resname, 1) else atomH
            elif c1 == "F":
                n = atomHf if fix_atom_name(name, resname, 1) else atomH
            elif c1 == "G":
                n = atomHg if fix_atom_name(name, resname, 1) else atomH
            elif c1 == "O":
                n = atomHo if fix_atom_name(name, resname, 1) else atomH
            else:
                n = atomH
        elif c0 == "I":
            n = {"N": atom_id["atomIn"], "R": atom_id["atomIr"]}.get(c1, -1)
        elif c0 == "K":
            if c1 == "R":
                n = atom_id["atomKr"]
        elif c0 == "L":
            n = {"A": atom_id["atomLa"], "I": atom_id["atomLi"], "U": atom_id["atomLu"]}.get(c1, -1)
        elif c0 == "M":
            n = {"D": atom_id["atomMd"], "G": atom_id["atomMg"], "N": atom_id["atomMn"], "O": atom_id["atomMo"]}.get(
                c1, -1
            )
        elif c0 == "N":
            n = {
                "A": atom_id["atomNa"],
                "B": atom_id["atomNb"],
                "C": atomC,
                "D": atom_id["atomNd"],
                "E": atom_id["atomNe"],
                "H": atomH,
                "I": atom_id["atomNi"],
                "N": atomN,
                "O": atomO,
                "P": atomP,
                "S": atomS,
            }.get(c1, atomN)
        elif c0 == "O":
            n = atom_id["atomOs"] if c1 == "S" else atomO
        elif c0 == "P":
            n = {
                "A": atom_id["atomPa"],
                "B": atom_id["atomPb"],
                "D": atom_id["atomPd"],
                "M": atom_id["atomPm"],
                "O": atom_id["atomPo"],
                "R": atom_id["atomPr"],
                "T": atom_id["atomPt"],
                "U": atom_id["atomPu"],
            }.get(c1, atomP)
        elif c0 == "R":
            n = {
                "A": atom_id["atomRa"],
                "B": atom_id["atomRb"],
                "E": atom_id["atomRe"],
                "H": atom_id["atomRh"],
                "N": atom_id["atomRn"],
                "U": atom_id["atomRu"],
            }.get(c1, -1)
        elif c0 == "S":
            n = {
                "B": atom_id["atomSb"],
                "C": atom_id["atomSc"],
                "E": atomSe,
                "I": atom_id["atomSi"],
                "M": atom_id["atomSm"],
                "N": atom_id["atomSn"],
                "R": atom_id["atomSr"],
            }.get(c1, atomS)
        elif c0 == "T":
            n = {
                "A": atom_id["atomTa"],
                "B": atom_id["atomTb"],
                "C": atom_id["atomTc"],
                "E": atom_id["atomTe"],
                "H": atom_id["atomTh"],
                "I": atom_id["atomTi"],
                "L": atom_id["atomTl"],
                "M": atom_id["atomTm"],
            }.get(c1, -1)
        elif c0 == "X":
            if c1 == "E":
                n = atom_id["atomXe"]
        elif c0 == "Y":
            if c1 == "B":
                n = atom_id["atomYb"]
        elif c0 == "Z":
            n = {"N": atom_id["atomZn"], "R": atom_id["atomZr"]}.get(c1, -1)
        if n < 0:
            n = atomC
            if c1 in ("H", "D"):
                n = atomH
            elif c1 == "C":
                n = atomC
            elif c1 == "N":
                n = atomN
            elif c1 == "O":
                n = atomO
            elif c1 == "P":
                n = atomP
            elif c1 == "S":
                n = atomS
            elif c1 == "I":
                n = atomI
            elif c1 == "K":
                n = atom_id["atomK"]
            elif c1 == "V":
                n = atom_id["atomV"]
            elif c1 == "W":
                n = atom_id["atomW"]
            elif c1 == "U":
                n = atom_id["atomU"]
            elif c1 == "A":
                n = {
                    "G": atom_id["atomAg"],
                    "L": atom_id["atomAl"],
                    "S": atom_id["atomAs"],
                    "U": atom_id["atomAu"],
                }.get(c2, atomC)
            elif c1 == "F" and c2 == "E":
                n = atom_id["atomFe"]
            elif c1 == "G" and c2 == "D":
                n = atom_id["atomGd"]
            elif c1 == "L" and c2 == "I":
                n = atom_id["atomLi"]
            elif c1 == "M":
                n = {"G": atom_id["atomMg"], "N": atom_id["atomMn"], "O": atom_id["atomMo"]}.get(c2, atomC)
            elif c1 == "Z" and c2 == "N":
                n = atom_id["atomZn"]
        return n

    @functools.lru_cache(maxsize=4096)
    def search_std_bonding_partner(resname, atomname, is_h=True):
        r = f"{resname:<3.3s}".upper()
        a = list(f"{atomname:<4.4s}".upper())
        if is_h:
            if a[0] == "D" and a[3] != " ":
                a[0] = "H"
            elif a[0] != "H" and a[1] == "D":
                a[1] = "H"
        a_str = "".join(a)
        q = f"{r}:{a_str}"
        p = stdconn.get(q)
        if p is None and r in na_alt_list:
            if r in ":GUA:GTP:GDP:GMP:GSP: DG: GR:":
                c = "G"
            elif r in ":ADE:ATP:ADP:AMP: DA: AR:":
                c = "A"
            elif r in ":CYT:CTP:CDP:CMP: DC: CR:":
                c = "C"
            elif r in ":URA:UTP:UDP:UMP: UR:":
                c = "U"
            elif r in ":THY:TTP:TDP:TMP: DT: TR:":
                c = "T"
            else:
                c = " "
            p = stdconn.get(f"  {c}:{a_str}")
        return p

    @functools.lru_cache(maxsize=4096)
    def is_carbonyl_atom(resname, atomname):
        return (
            (resname in AAList and atomname == " C  ")
            or (resname in ":ASP:ASN:ASX:" and atomname in ": CG :")
            or (resname in ":GLU:GLN:GLX:" and atomname in ": CD :")
        )

    @functools.lru_cache(maxsize=4096)
    def set_properties(elem, atomname, resname, is_het):
        props = 0
        if resname in AromaticAAList or resname in NAList:
            for rlist, alist, bits in AromaticAtomsTbl:
                if resname in rlist and atomname in alist:
                    props |= AROMATIC_PROP | bits
                    if atno_arr[elem] == 1:
                        elem = atomHarom
                    if atno_arr[elem] == 6:
                        elem = atomCarom
                    break
        if resname in MethylResList:
            for rlist, alist, _ in MethylAtomsTbl:
                if resname in rlist and atomname in alist:
                    props |= METHYL_PROP
                    break
        if resname in MethyleneResList:
            for rlist, alist, _ in MethyleneAtomsTbl:
                if resname in rlist and atomname in alist:
                    props |= METHYLENE_PROP
                    break
        if resname in WaterList:
            props |= WATER_PROP
        elif is_het and resname not in ":MSE:":
            props |= HET_PROP

        if resname in AAList:
            props |= PROT_PROP
            if resname in HphobicAAList:
                props |= RHPHOBIC_PROP
            elif resname in HphilicAAList:
                props |= RHPHILIC_PROP
            elif resname in ChargedAAList:
                props |= RCHARGED_PROP
            if elem == atomC or atno_arr[elem] == 1:
                if atomname[2] == "A":
                    if atomname == "2HA ":
                        props |= ALPHA_PROP
                    else:
                        props |= ALPHA_PROP | MC_PROP
                elif atomname[2] == "B":
                    props |= BETA_PROP
                elif atomname == " C  ":
                    props |= MC_PROP
                elif atomname[1:] == "H  ":
                    props |= MC_PROP
                elif atomname[1:] == "D  ":
                    props |= MC_PROP
                elif atomname in ": HXT: DXT: HN :":
                    props |= MC_PROP
            elif atomname in (" N  ", " O  "):
                props |= MC_PROP
            elif atomname in AlwaysChargedAtomsList:
                props |= MC_PROP
            if not (props & MC_PROP):
                props |= SC_PROP
        elif resname in NAList:
            props |= DNA_PROP
            if atomname in NAbackboneList:
                props |= MC_PROP
            else:
                props |= SC_PROP

        if elem == atomN:
            props |= N_PROP
        elif elem == atomC:
            props |= C_PROP
        elif elem == atomO:
            props |= O_PROP
        elif elem == atomS:
            props |= S_PROP
        elif elem == atomP:
            props |= P_PROP
        if atno_arr[elem] == 1:
            props |= H_PROP
        if elem_flags_arr[elem] & 1:
            props |= METAL_PROP
        if elem_flags_arr[elem] & 2:
            props |= ION_PROP

        if props & TEST_ACCEPT_ANGLE_PROP:
            props |= ACCEPTOR_PROP

        if props & WATER_PROP:
            if elem == atomO:
                props |= DONOR_PROP | ACCEPTOR_PROP
            elif atno_arr[elem] == 1:
                props |= DONOR_PROP
        elif elem in (atomN, atomO, atomS) or atno_arr[elem] == 1:
            if resname in AAList or resname in NAList or resname in WaterList:
                for rlist, alist, bits in DonorAcceptorAtomTbl:
                    if resname in rlist and atomname in alist:
                        if not (bits & CH_DONOR_PROP):
                            props |= bits
                        break
                if not (atno_arr[elem] == 1 or (props & (ACCEPTOR_PROP | DONOR_PROP))):
                    props |= ACCEPTOR_PROP
            elif atno_arr[elem] != 1:
                props |= DONOR_PROP | ACCEPTOR_PROP

        add_charge = False
        check_terminal = False
        if props & PROT_PROP:
            if atomname in AmbigChargedAtomsList:
                if atomname == " N  ":
                    props |= MAYBECHG_PROP
                elif atomname == " O  ":
                    props |= MAYBECHG_PROP
                    check_terminal = True
            elif atomname in AlwaysChargedAtomsList:
                add_charge = True
                check_terminal = True
            elif resname in ChargedAAList:
                for rlist, alist, bits in ChargedAAAtomsTbl:
                    if resname in rlist and atomname in alist:
                        props |= bits
                        break
        elif (props & DNA_PROP) and (atomname in ChargedNucAcidAtomsList):
            add_charge = True
        elif props & METAL_PROP:
            add_charge = True
        elif props & ION_PROP:
            add_charge = True

        if add_charge:
            if props & METAL_PROP:
                props |= POSITIVE_PROP
            elif props & H_PROP:
                props |= POSITIVE_PROP
            elif elem == atomN:
                props |= POSITIVE_PROP
            elif elem in (atomO, atomS, atomP):
                props |= NEGATIVE_PROP
            elif elem in (atomF, atomCl, atomBr, atomI):
                props |= NEGATIVE_PROP | ACCEPTOR_PROP

        if check_terminal:
            props |= CHECK_ENDS_PROP
        return elem, props

    atoms = []
    previd = 0
    rescnt = 0
    model = 0
    prev_icode = " "
    prev_chain = "??"

    ambigN = [None] * 8
    ambigO = [None] * 8
    res_mc_oxy_cnt = 0
    first_res = 0
    last_res = 0
    nt_markers = 0
    ct_markers = 0
    his_nh = [0, 0, 0, 0]

    def init_end_data():
        nonlocal ambigN, ambigO, res_mc_oxy_cnt, first_res, last_res, nt_markers, ct_markers, his_nh
        ambigN = [None] * 8
        ambigO = [None] * 8
        res_mc_oxy_cnt = 0
        first_res = 0
        last_res = 0
        nt_markers = 0
        ct_markers = 0
        his_nh = [0, 0, 0, 0]

    def his_nh_check(rc):
        nonlocal his_nh
        if his_nh[0]:
            for idx in range(len(atoms) - 1, -1, -1):
                rat = atoms[idx]
                if rat["rescnt"] < rc:
                    break
                if rat["rescnt"] == rc and (rat["props"] & POSITIVE_PROP):
                    altc = rat["altloc"].upper()
                    ialt = 0
                    if altc in (" ", "A", "1"):
                        ialt = 1
                    elif altc in ("B", "2"):
                        ialt = 2
                    elif altc in ("C", "3"):
                        ialt = 3
                    if ialt > 0:
                        if his_nh[ialt] < 2:
                            rat["props"] &= ~POSITIVE_PROP
                        else:
                            rat["props"] &= ~ACCEPTOR_PROP
            his_nh = [0, 0, 0, 0]

    def cterm_check(rc, is_chain_end):
        nonlocal last_res
        if is_chain_end and rc:
            last_res = rc
            for i in range(4):
                a = ambigN[i]
                if a is not None:
                    if a["rescnt"] == first_res and nt_markers == first_res:
                        a["props"] |= POSITIVE_PROP
                else:
                    break
            for i in range(8):
                a = ambigO[i]
                if a is not None:
                    if a["rescnt"] == last_res and ct_markers == last_res:
                        a["props"] |= NEGATIVE_PROP
                else:
                    break
            init_end_data()

    def nterm_check(rc, is_chain_end):
        nonlocal res_mc_oxy_cnt, ambigO, ambigN, first_res
        res_mc_oxy_cnt = 0
        for i in range(8):
            ambigO[i] = None
        for i in range(4, 8):
            ambigN[i] = None
        if is_chain_end:
            first_res = rc

    def noticed_nt(rc):
        for i in range(4, 8):
            a = ambigN[i]
            if a is not None:
                if a["rescnt"] == rc:
                    a["props"] |= POSITIVE_PROP
            else:
                break

    def noticed_ct(rc):
        for i in range(8):
            a = ambigO[i]
            if a is not None:
                if a["rescnt"] == rc:
                    a["props"] |= NEGATIVE_PROP
            else:
                break

    def process_res_info(a):
        nonlocal res_mc_oxy_cnt, nt_markers, ct_markers
        if a["resname"] in "HIS:his":
            his_nh[0] = 1
            if a["atomname"] in " HD1: hd1: HE2: he2":
                altc = a["altloc"].upper()
                if altc in (" ", "A", "1"):
                    his_nh[1] += 1
                elif altc in ("B", "2"):
                    his_nh[2] += 1
                elif altc in ("C", "3"):
                    his_nh[3] += 1
        if a["props"] & CHECK_ENDS_PROP:
            if atno_arr[a["elem"]] == 1:
                if not nt_markers:
                    nt_markers = a["rescnt"]
                noticed_nt(a["rescnt"])
            elif a["elem"] == atomO and a["altloc"] in (" ", "A", "1"):
                res_mc_oxy_cnt += 1
                if res_mc_oxy_cnt == 2:
                    ct_markers = a["rescnt"]
                    noticed_ct(a["rescnt"])
            elif "OXT" in a["atomname"]:
                ct_markers = a["rescnt"]
                noticed_ct(a["rescnt"])
        if a["props"] & MAYBECHG_PROP:
            if a["atomname"] == " N  ":
                if first_res == a["rescnt"]:
                    a["props"] |= CHECK_ENDS_PROP
                for i in range(4):
                    if ambigN[i] is None:
                        if i == 0 or ambigN[i - 1]["res_obj_id"] == a["res_obj_id"]:
                            ambigN[i] = a
                        break
                for i in range(4, 8):
                    if ambigN[i] is None:
                        if i == 4 or ambigN[i - 1]["res_obj_id"] == a["res_obj_id"]:
                            ambigN[i] = a
                        break
            if a["atomname"] == " O  ":
                for i in range(8):
                    if ambigO[i] is None:
                        if i == 0 or ambigO[i - 1]["res_obj_id"] == a["res_obj_id"]:
                            ambigO[i] = a
                        break

    prev_res_tuple = None
    cur_res_obj_id = 0
    cur_res_name = ""

    with open(pdb_path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")[:80]
            if not line:
                continue
            rec6 = line[:6].upper()
            if rec6.startswith("TER") and len(line) >= 27:
                prev_chain = "??"
                continue
            if rec6.startswith("MODEL") and len(line) >= 14:
                try:
                    model = int(line[6:14].strip())
                except ValueError:
                    model = 0
                continue
            is_atom = rec6.startswith("ATOM") and len(line) >= 47
            is_het = rec6.startswith("HETA") and len(line) >= 47
            if not (is_atom or is_het):
                continue
            if line[13].upper() == "Q":
                continue
            atomname = line[12:16].upper()
            altloc = line[16]
            resname = line[17:20].upper()
            chain = line[20:22]
            hy36 = line[22:26]
            resid = hy36decode4(hy36)
            icode = line[26]
            segid = (line[72:76] if len(line) >= 76 else line[72:].ljust(4)) if len(line) >= 73 else "    "
            if segid != "    " and chain == "  ":
                chain = segid

            if resname in ":ASX:GLX:ASN:GLN:" and atomname[1] == "A":
                elem = atomO
            else:
                elem = identify_atom(atomname, resname)

            bondedto = search_std_bonding_partner(resname, atomname, is_h=True) if atno_arr[elem] == 1 else None
            elem, props = set_properties(elem, atomname, resname, is_het)

            x = float(np.float32(float(line[30:38].strip() or 0.0)))
            y = float(np.float32(float(line[38:46].strip() or 0.0)))
            z = float(np.float32(float(line[46:54].strip() or 0.0)))
            occ = float(np.float32(float(line[54:60].strip()) if len(line) >= 55 and line[54:60].strip() else 1.0))
            bval = float(np.float32(float(line[60:66].strip()) if len(line) >= 61 and line[60:66].strip() else 0.0))

            is_co = is_carbonyl_atom(resname, atomname)
            rad = co_rad if is_co else float(erad_arr[elem])
            cov_r = float(covrad_arr[elem])

            if resid != previd or icode != prev_icode or chain != prev_chain:
                if rescnt:
                    his_nh_check(rescnt)
                cterm_check(rescnt, chain != prev_chain)
                rescnt += 1
                nterm_check(rescnt, chain != prev_chain)
                previd = resid
                prev_icode = icode
                prev_chain = chain

            res_tuple = (resid, icode, chain, model)
            is_dummy = x >= 9998.0 and y >= 9998.0 and z >= 9998.0
            a_dict = {
                "atomname": atomname,
                "altloc": altloc,
                "resname": resname,
                "chain": chain,
                "hy36": hy36,
                "resid": resid,
                "icode": icode,
                "segid": segid,
                "model": model,
                "rescnt": rescnt,
                "res_obj_id": cur_res_obj_id if res_tuple == prev_res_tuple else cur_res_obj_id + 1,
                "elem": elem,
                "is_co": is_co,
                "props": props,
                "bondedto": bondedto,
                "x": x,
                "y": y,
                "z": z,
                "occ": occ,
                "bval": bval,
                "radius": rad,
                "covRad": cov_r,
                "flags": 0,
            }
            process_res_info(a_dict)
            if is_dummy:
                continue
            if res_tuple != prev_res_tuple:
                cur_res_obj_id += 1
                prev_res_tuple = res_tuple
                cur_res_name = resname
            a_dict["res_obj_id"] = cur_res_obj_id
            # In probe.c:2376, a->r = *reslstptr shares the first atom's residue block
            a_dict["resname"] = cur_res_name
            a_dict["is_co"] = is_carbonyl_atom(cur_res_name, atomname)
            atoms.append(a_dict)

    if rescnt:
        his_nh_check(rescnt)
    cterm_check(rescnt, True)

    # Reverse atoms to match loadAtoms head-insertion order (allMainAtoms)
    atoms.reverse()

    # selectSource: SET1 = "ogt<ogt> not water", SET2 = "ogt<ogt>"
    occ_thresh = ogt / 100.0
    for a in atoms:
        if a["occ"] > occ_thresh:
            a["flags"] |= 2
            if not (a["props"] & WATER_PROP):
                a["flags"] |= 1

    # updateHydrogenInfo:
    # 1. H atoms only bond to heavy atoms in the same residue (res_obj_id)
    # 2. Unprotonated water oxygens query nearby acceptors via lazy cKDTree
    res_heavies = {}
    has_sel = False
    for a in atoms:
        if a["flags"] & 3:
            has_sel = True
            if atno_arr[a["elem"]] != 1:
                res_heavies.setdefault(a["res_obj_id"], []).append(a)
    if not has_sel:
        return atoms

    water_tree = None
    sel_indices = None
    max_sel_rad = 0.0

    new_atoms = []
    for idx, src in enumerate(atoms):
        new_atoms.append(src)
        e = src["elem"]
        if atno_arr[e] == 1:
            if src["flags"] & 3:
                parent_type = e
                bonded_heavies = []
                s_alt = src["altloc"]
                s_mod = src["model"]
                for nb in res_heavies.get(src["res_obj_id"], ()):
                    if nb["model"] != s_mod:
                        continue
                    n_alt = nb["altloc"]
                    if s_alt != n_alt and s_alt != " " and n_alt != " ":
                        continue
                    dx = src["x"] - nb["x"]
                    dy = src["y"] - nb["y"]
                    dz = src["z"] - nb["z"]
                    dsq = float(np.float32(dx * dx + dy * dy + dz * dz))
                    lim = float(np.float32(src["radius"]) + np.float32(nb["radius"]))
                    if not (0.00001 < dsq <= lim * lim):
                        continue
                    b_lim = float(np.float32(src["covRad"]) + np.float32(nb["covRad"]) + COVRADFUDGE)
                    t_lim = float(np.float32(src["covRad"]) + np.float32(nb["covRad"]) - np.float32(0.6))
                    if dsq <= b_lim * b_lim and not (dsq <= t_lim * t_lim):
                        if src["bondedto"] is None or nb["atomname"] in src["bondedto"]:
                            bonded_heavies.append(nb)
                if bonded_heavies:
                    parent_type = bonded_heavies[0]["elem"]
                if parent_type in (atomN, atomO, atomS):
                    src["props"] |= DONOR_PROP
                    src["elem"] = atomHpolar
                    src["radius"] = float(erad_arr[atomHpolar])
                    if (abs(src["occ"]) < 0.25 or src["bval"] >= 80.0) and (src["props"] & WATER_PROP):
                        src["elem"] = ignoreAtom
                        src["radius"] = 0.0
                    if abs(src["occ"]) >= 0.25 and src["bval"] < 80.0:
                        for nb in bonded_heavies:
                            nb["props"] &= ~DONOR_PROP
        elif (src["flags"] & 3) and (src["props"] & WATER_PROP) and e == atomO:
            if src["props"] & DONOR_PROP:
                for j in range(idx + 1, len(atoms)):
                    nb = atoms[j]
                    if nb["res_obj_id"] != src["res_obj_id"]:
                        break
                    if atno_arr[nb["elem"]] == 1 and nb["altloc"] == src["altloc"] and nb["occ"] > 0.1:
                        src["props"] &= ~DONOR_PROP
            if src["props"] & DONOR_PROP:
                src["props"] |= AMBIGWATER_PROP | ACCEPTOR_PROP
                src["props"] &= ~DONOR_PROP
                if water_tree is None:
                    sel_indices = [i for i, a in enumerate(atoms) if a["flags"] & 3]
                    coords = np.array(
                        [[atoms[i]["x"], atoms[i]["y"], atoms[i]["z"]] for i in sel_indices],
                        dtype=np.float64,
                    )
                    water_tree = cKDTree(coords)
                    max_sel_rad = max(atoms[i]["radius"] for i in sel_indices)
                # Build phantom H? atoms toward nearby acceptors (probe.c:6190-6320)
                prox = []
                s_alt = src["altloc"]
                s_mod = src["model"]
                hit_p = water_tree.query_ball_point(
                    [src["x"], src["y"], src["z"]],
                    r=src["radius"] + max_sel_rad + 0.501,
                )
                for p_i in sorted(hit_p):
                    nb_i = sel_indices[p_i]
                    if nb_i == idx:
                        continue
                    nb = atoms[nb_i]
                    if nb["model"] != s_mod:
                        continue
                    n_alt = nb["altloc"]
                    if s_alt != n_alt and s_alt != " " and n_alt != " ":
                        continue
                    dx = nb["x"] - src["x"]
                    dy = nb["y"] - src["y"]
                    dz = nb["z"] - src["z"]
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    dsq = float(np.float32(dx * dx + dy * dy + dz * dz))
                    lim = float(np.float32(src["radius"]) + np.float32(nb["radius"]) + np.float32(0.5))
                    if not (0.00001 < dsq <= lim * lim):
                        continue
                    if nb["props"] & ACCEPTOR_PROP:
                        hbgap = float(np.float32(dist - (nb["radius"] + 2.0)))
                        found_arom = False
                        if nb["props"] & AROMATIC_PROP:
                            for p_entry in prox:
                                if (
                                    nb["res_obj_id"] == p_entry[0]["res_obj_id"]
                                    and (p_entry[0]["props"] & AROMATIC_PROP)
                                ):
                                    if hbgap < p_entry[1]:
                                        p_entry[0] = nb
                                        p_entry[1] = hbgap
                                        p_entry[2] = (dx, dy, dz, dist)
                                    found_arom = True
                                    break
                        if not found_arom and len(prox) < 20:
                            prox.append([nb, hbgap, (dx, dy, dz, dist)])
                for nb, hbgap, (dx, dy, dz, dist) in prox:
                    if (nb["props"] & ACCEPTOR_PROP) and dist > 0.0:
                        bond_len = 1.0 + max(-1.0, min(0.0, hbgap + 0.6))
                        scale = bond_len / dist
                        new_h = dict(src)
                        new_h["x"] = src["x"] + dx * scale
                        new_h["y"] = src["y"] + dy * scale
                        new_h["z"] = src["z"] + dz * scale
                        new_h["elem"] = atomHOd
                        new_h["is_co"] = False
                        new_h["radius"] = float(erad_arr[atomHOd])
                        new_h["covRad"] = float(covrad_arr[atomHOd])
                        new_h["atomname"] = " H? "
                        new_h["props"] = (src["props"] | DONOR_PROP) & ~ACCEPTOR_PROP
                        new_atoms.append(new_h)
        elif (src["flags"] & 3) and not (src["props"] & WATER_PROP) and e in (atomN, atomO, atomS):
            src["props"] &= ~DONOR_PROP

    return new_atoms


@lazy_jit(static_argnames=("ignore_atom", "atom_hod", "donor_prop", "acceptor_prop", "pos_prop", "neg_prop", "water_prop"))
def _examine_dots_batch_kernel(
    src_xyz,          # (B, 3) float64
    src_rad,          # (B,) float32
    src_cov,          # (B,) float32
    src_elem,         # (B,) int32
    src_is_h,         # (B,) bool
    src_res_id,       # (B,) int32
    src_props,        # (B,) int32
    src_occ,          # (B,) float32
    src_alt_blank,    # (B,) bool
    src_h_allow,      # (B, 2) uint64
    src_hv_bit,       # (B, 2) uint64
    src_dot_idx,      # (B,) int32
    nb_valid,         # (B, K) bool
    nb_xyz,           # (B, K, 3) float64
    nb_rad,           # (B, K) float32
    nb_cov,           # (B, K) float32
    nb_elem,          # (B, K) int32
    nb_is_h,          # (B, K) bool
    nb_res_id,        # (B, K) int32
    nb_props,         # (B, K) int32
    nb_flags,         # (B, K) int32
    nb_occ,           # (B, K) float32
    nb_alt_blank,     # (B, K) bool
    nb_h_allow,       # (B, K, 2) uint64
    nb_hv_bit,        # (B, K, 2) uint64
    dot_vecs_tbl,     # (T, D, 3) float64
    exp_vecs_tbl,     # (T, D, 3) float64
    dot_mask_tbl,     # (T, D) bool
    ignore_atom,
    atom_hod,
    donor_prop,
    acceptor_prop,
    pos_prop,
    neg_prop,
    water_prop,
):
    """
    GPU kernel executing:
      1. 4-bond covalent BFS (`markBonds` + `fixupLongBondChains(cutoff=3)`) on `(B, K)` neighbors.
      2. Dot-sphere surface intersection (`examineDots`) on `(B, D, K)`.
      3. Per-`(src, neighbor_k)` condensation of `hb` flag and minimum `bo` clash gap (`<= -0.4`).
    """
    _, jnp = _get_jax()
    B, K = nb_valid.shape

    # --- 1. 4-bond covalent BFS (markBonds + fixupLongBondChains) ---
    # Source-to-neighbor bonds (B, K)
    dsq_sn = jnp.float32(jnp.sum((src_xyz[:, None, :] - nb_xyz) ** 2, axis=-1))
    lim_sn = src_cov[:, None] + nb_cov + jnp.float32(0.2)
    close_sn = src_cov[:, None] + nb_cov - jnp.float32(0.6)
    dist_ok_sn = nb_valid & (dsq_sn <= (lim_sn * lim_sn)) & (dsq_sn > (close_sn * close_sn))

    both_hv_sn = (~src_is_h[:, None]) & (~nb_is_h)
    one_h_sn = (src_is_h[:, None] ^ nb_is_h) & (src_res_id[:, None] == nb_res_id)
    h_allow_sn_0 = jnp.where(src_is_h[:, None], src_h_allow[:, None, 0], nb_h_allow[:, :, 0])
    h_allow_sn_1 = jnp.where(src_is_h[:, None], src_h_allow[:, None, 1], nb_h_allow[:, :, 1])
    hv_bit_sn_0 = jnp.where(src_is_h[:, None], nb_hv_bit[:, :, 0], src_hv_bit[:, None, 0])
    hv_bit_sn_1 = jnp.where(src_is_h[:, None], nb_hv_bit[:, :, 1], src_hv_bit[:, None, 1])
    parent_ok_sn = ((h_allow_sn_0 & hv_bit_sn_0) | (h_allow_sn_1 & hv_bit_sn_1)) != jnp.uint64(0)
    bond_sn = dist_ok_sn & (both_hv_sn | (one_h_sn & parent_ok_sn))

    # Neighbor-to-neighbor bonds (B, K, K)
    dsq_nn = jnp.float32(jnp.sum((nb_xyz[:, :, None, :] - nb_xyz[:, None, :, :]) ** 2, axis=-1))
    lim_nn = nb_cov[:, :, None] + nb_cov[:, None, :] + jnp.float32(0.2)
    close_nn = nb_cov[:, :, None] + nb_cov[:, None, :] - jnp.float32(0.6)
    valid_nn = nb_valid[:, :, None] & nb_valid[:, None, :] & (~jnp.eye(K, dtype=bool)[None, :, :])
    dist_ok_nn = valid_nn & (dsq_nn <= (lim_nn * lim_nn)) & (dsq_nn > (close_nn * close_nn))

    both_hv_nn = (~nb_is_h[:, :, None]) & (~nb_is_h[:, None, :])
    one_h_nn = (nb_is_h[:, :, None] ^ nb_is_h[:, None, :]) & (nb_res_id[:, :, None] == nb_res_id[:, None, :])
    h_allow_nn_0 = jnp.where(nb_is_h[:, :, None], nb_h_allow[:, :, None, 0], nb_h_allow[:, None, :, 0])
    h_allow_nn_1 = jnp.where(nb_is_h[:, :, None], nb_h_allow[:, :, None, 1], nb_h_allow[:, None, :, 1])
    hv_bit_nn_0 = jnp.where(nb_is_h[:, :, None], nb_hv_bit[:, None, :, 0], nb_hv_bit[:, :, None, 0])
    hv_bit_nn_1 = jnp.where(nb_is_h[:, :, None], nb_hv_bit[:, None, :, 1], nb_hv_bit[:, :, None, 1])
    parent_ok_nn = ((h_allow_nn_0 & hv_bit_nn_0) | (h_allow_nn_1 & hv_bit_nn_1)) != jnp.uint64(0)
    bond_nn = dist_ok_nn & (both_hv_nn | (one_h_nn & parent_ok_nn))

    r1 = bond_sn
    r2 = jnp.any(r1[:, :, None] & bond_nn, axis=1)
    r3 = jnp.any(r2[:, :, None] & bond_nn, axis=1)
    r4 = jnp.any(r3[:, :, None] & bond_nn, axis=1)
    r123 = r1 | r2 | r3
    is_bonded = jnp.where(both_hv_sn, r123, r123 | r4)

    # --- 2. Unbonded candidate & H-bond masks (B, K) ---
    src_is_don = (src_props & donor_prop) != 0
    src_is_acc = (src_props & acceptor_prop) != 0
    mask_hb = jnp.where(
        src_is_don,
        jnp.where(src_is_acc, acceptor_prop | donor_prop, acceptor_prop),
        jnp.where(src_is_acc, donor_prop, 0),
    )
    chg_mask = pos_prop | neg_prop
    both_charged = ((src_props[:, None] & chg_mask) != 0) & ((nb_props & chg_mask) != 0)
    charge_comp = (
        (((src_props[:, None] & pos_prop) != 0) & ((nb_props & neg_prop) != 0))
        | (((src_props[:, None] & neg_prop) != 0) & ((nb_props & pos_prop) != 0))
    )
    is_hb_pair = ((nb_props & mask_hb[:, None]) != 0) & ((~both_charged) | charge_comp)
    hb_cutoff_pair = jnp.where(both_charged, jnp.float32(0.8), jnp.float32(0.6))

    wat_wat = ((src_props[:, None] & water_prop) != 0) & ((nb_props & water_prop) != 0)
    hod_involved = (src_elem[:, None] == atom_hod) | (nb_elem == atom_hod)

    cand_unbonded = (
        nb_valid
        & (~is_bonded)
        & (nb_elem != ignore_atom)
        & (nb_occ >= jnp.float32(0.02))
        & (~wat_wat)
        & ((~hod_involved) | is_hb_pair)
    )
    is_occluder = nb_valid & is_bonded & (nb_elem != atom_hod)

    # --- 3. Dot-sphere intersection (B, D, K) ---
    dot_vecs = dot_vecs_tbl[src_dot_idx]   # (B, D, 3)
    exp_vecs = exp_vecs_tbl[src_dot_idx]   # (B, D, 3)
    dot_valid = dot_mask_tbl[src_dot_idx]  # (B, D)

    dotloc = src_xyz[:, None, :] + dot_vecs  # (B, D, 3)
    exploc = src_xyz[:, None, :] + exp_vecs  # (B, D, 3)

    exp_dsq = jnp.sum((exploc[:, :, None, :] - nb_xyz[:, None, :, :]) ** 2, axis=-1)
    exp_lim = jnp.float64(jnp.float32(0.25) + nb_rad[:, None, :])
    nearpt = (exp_dsq <= (exp_lim * exp_lim)) & cand_unbonded[:, None, :]

    dot_dsq = jnp.sum((dotloc[:, :, None, :] - nb_xyz[:, None, :, :]) ** 2, axis=-1)
    gap = jnp.float32(jnp.sqrt(dot_dsq) - jnp.float64(nb_rad[:, None, :]))
    gap_masked = jnp.where(nearpt, gap, jnp.float32(999.9))

    min_gap = jnp.min(gap_masked, axis=-1)       # (B, D)
    cause_k = jnp.argmin(gap_masked, axis=-1)    # (B, D)
    ok = (min_gap < jnp.float32(999.9)) & dot_valid

    # Target SET2 check
    cause_in_set2 = jnp.take_along_axis((nb_flags & 2) != 0, cause_k, axis=-1)
    ok = ok & cause_in_set2

    # Bonded-atom occlusion check
    occ_rad_sq = jnp.float64(nb_rad[:, None, :] * nb_rad[:, None, :])
    occluded = jnp.any(is_occluder[:, None, :] & (dot_dsq <= occ_rad_sq), axis=-1)
    ok = ok & (~occluded)

    # Classify surviving dots
    cause_is_hb = jnp.take_along_axis(is_hb_pair, cause_k, axis=-1)
    cause_hb_cut = jnp.take_along_axis(hb_cutoff_pair, cause_k, axis=-1)
    cause_elem = jnp.take_along_axis(nb_elem, cause_k, axis=-1)
    cause_alt_blank = jnp.take_along_axis(nb_alt_blank, cause_k, axis=-1)
    cause_occ = jnp.take_along_axis(nb_occ, cause_k, axis=-1)

    is_contact = min_gap > jnp.float32(0.0)
    too_close_hb = (~is_contact) & cause_is_hb & (min_gap < -cause_hb_cut)
    is_hb_dot = ok & (~is_contact) & cause_is_hb & (~too_close_hb)
    is_bump_dot = (
        ok
        & (~is_contact)
        & ((~cause_is_hb) | too_close_hb)
        & (src_elem[:, None] != atom_hod)
        & (cause_elem != atom_hod)
    )
    adj_gap = jnp.where(too_close_hb, min_gap + cause_hb_cut, min_gap)

    # Altloc/occupancy post-filter (probe.c:4489)
    occ_filtered = (src_alt_blank[:, None] | cause_alt_blank) & ((src_occ[:, None] + cause_occ) <= jnp.float32(1.0))
    is_bo_dot = is_bump_dot & (~occ_filtered) & (adj_gap <= jnp.float32(-0.4))

    # --- 4. Condense over D dots to (B, K) ---
    k_grid = jnp.arange(K, dtype=cause_k.dtype)[None, None, :]  # (1, 1, K)
    match_k = cause_k[:, :, None] == k_grid                    # (B, D, K)
    pair_has_hb = jnp.any(is_hb_dot[:, :, None] & match_k, axis=1)
    pair_min_bo_gap = jnp.min(
        jnp.where(is_bo_dot[:, :, None] & match_k, adj_gap[:, :, None], jnp.float32(999.9)),
        axis=1,
    )
    return pair_has_hb, pair_min_bo_gap


@register_gpu_op("probe_dots")
def _gpu_probe_dots(
    xyz,
    rad,
    cov,
    elem,
    is_h,
    res_id,
    props,
    flags,
    occ,
    alt_blank,
    h_allow,
    hv_bit,
    dot_idx,
    unique_src,
    starts,
    counts,
    dir_nb,
    batch_size=2048,
):
    _, jnp = _get_jax()
    tbl = load_probe_tables()
    atom_id = tbl["atom_id"]
    props_map = tbl["props_map"]
    dot_counts = tbl["dot_counts"]

    ignore_atom = int(atom_id["ignoreAtom"])
    atom_hod = int(atom_id["atomHOd"])
    donor_prop = int(props_map["DONOR_PROP"])
    acceptor_prop = int(props_map["ACCEPTOR_PROP"])
    pos_prop = int(props_map["POSITIVE_PROP"])
    neg_prop = int(props_map["NEGATIVE_PROP"])
    water_prop = int(props_map["WATER_PROP"])

    raw_k_max = int(np.max(counts))
    k_pad = max(32, 1 << (raw_k_max - 1).bit_length())

    src_d_counts = dot_counts[dot_idx[unique_src]]
    tier_limits = (320, 656, 1216)

    hb_s_list, hb_t_list = [], []
    bo_s_list, bo_t_list, bo_g_list = [], [], []

    for tier_idx, max_d in enumerate(tier_limits):
        low_d = 0 if tier_idx == 0 else tier_limits[tier_idx - 1]
        in_tier = (src_d_counts > low_d) & (src_d_counts <= max_d)
        if not np.any(in_tier):
            continue
        tier_src = unique_src[in_tier]
        tier_starts = starts[in_tier]
        tier_counts = counts[in_tier]
        dot_vecs_tbl, exp_vecs_tbl, dot_mask_tbl = tbl["tiers"][max_d]

        n_tier = len(tier_src)
        for b_start in range(0, n_tier, batch_size):
            b_end = min(b_start + batch_size, n_tier)
            b_len = b_end - b_start
            b_pad = min(batch_size, max(256, 1 << (b_len - 1).bit_length()))

            b_src = np.zeros(b_pad, dtype=np.int32)
            b_src[:b_len] = tier_src[b_start:b_end]
            b_starts = tier_starts[b_start:b_end]
            b_counts = tier_counts[b_start:b_end]

            nb_mat = np.zeros((b_pad, k_pad), dtype=np.int32)
            nb_val = np.zeros((b_pad, k_pad), dtype=bool)
            for r_i in range(b_len):
                c_i = b_counts[r_i]
                s_i = b_starts[r_i]
                nb_mat[r_i, :c_i] = dir_nb[s_i : s_i + c_i]
                nb_val[r_i, :c_i] = True

            pair_hb, pair_bo_gap = _examine_dots_batch_kernel(
                jnp.asarray(xyz[b_src]),
                jnp.asarray(rad[b_src]),
                jnp.asarray(cov[b_src]),
                jnp.asarray(elem[b_src]),
                jnp.asarray(is_h[b_src]),
                jnp.asarray(res_id[b_src]),
                jnp.asarray(props[b_src]),
                jnp.asarray(occ[b_src]),
                jnp.asarray(alt_blank[b_src]),
                jnp.asarray(h_allow[b_src]),
                jnp.asarray(hv_bit[b_src]),
                jnp.asarray(dot_idx[b_src]),
                jnp.asarray(nb_val),
                jnp.asarray(xyz[nb_mat]),
                jnp.asarray(rad[nb_mat]),
                jnp.asarray(cov[nb_mat]),
                jnp.asarray(elem[nb_mat]),
                jnp.asarray(is_h[nb_mat]),
                jnp.asarray(res_id[nb_mat]),
                jnp.asarray(props[nb_mat]),
                jnp.asarray(flags[nb_mat]),
                jnp.asarray(occ[nb_mat]),
                jnp.asarray(alt_blank[nb_mat]),
                jnp.asarray(h_allow[nb_mat]),
                jnp.asarray(hv_bit[nb_mat]),
                dot_vecs_tbl,
                exp_vecs_tbl,
                dot_mask_tbl,
                ignore_atom=ignore_atom,
                atom_hod=atom_hod,
                donor_prop=donor_prop,
                acceptor_prop=acceptor_prop,
                pos_prop=pos_prop,
                neg_prop=neg_prop,
                water_prop=water_prop,
            )
            hb_np = np.asarray(pair_hb)
            bo_np = np.asarray(pair_bo_gap)

            hb_rows, hb_cols = np.nonzero(hb_np)
            if len(hb_rows):
                hb_s_list.append(b_src[hb_rows])
                hb_t_list.append(nb_mat[hb_rows, hb_cols])

            bo_rows, bo_cols = np.nonzero(bo_np <= -0.4)
            if len(bo_rows):
                bo_s_list.append(b_src[bo_rows])
                bo_t_list.append(nb_mat[bo_rows, bo_cols])
                bo_g_list.append(bo_np[bo_rows, bo_cols])

    hb_s = np.concatenate(hb_s_list) if hb_s_list else np.empty(0, dtype=np.int32)
    hb_t = np.concatenate(hb_t_list) if hb_t_list else np.empty(0, dtype=np.int32)
    bo_s = np.concatenate(bo_s_list) if bo_s_list else np.empty(0, dtype=np.int32)
    bo_t = np.concatenate(bo_t_list) if bo_t_list else np.empty(0, dtype=np.int32)
    bo_g = np.concatenate(bo_g_list) if bo_g_list else np.empty(0, dtype=np.float32)
    return hb_s, hb_t, bo_s, bo_t, bo_g


def probe_clashes_jax(withh_pdb, occupancy_max=10.0, use_segids=False, model_id="", batch_size=2048):
    """
    Run GPU all-atom clash detection on `withh_pdb`, returning the list of `ProbeClash`
    objects matching `mmtbx.validation.clashscore.probe_clashscore_manager.process_raw_probe_output`.
    """
    ogt = 10
    if occupancy_max < ogt:
        ogt = int(occupancy_max)

    tbl = load_cpu_probe_tables()
    atom_id = tbl["atom_id"]
    atno_arr = tbl["atno"]
    co_dot_idx = tbl["co_dot_idx"]
    heavy_name_to_bits = tbl["heavy_name_to_bits"]
    default_heavy_bits = tbl["default_heavy_bits"]
    all_heavy_bits = tbl["all_heavy_bits"]
    bondedto_str_to_bits = tbl["bondedto_str_to_bits"]

    ignore_atom = int(atom_id["ignoreAtom"])

    atoms = load_probe_atoms(withh_pdb, ogt=ogt)
    n_atoms = len(atoms)
    if n_atoms == 0:
        return []

    # Convert atom list to contiguous NumPy arrays
    xyz = np.empty((n_atoms, 3), dtype=np.float64)
    rad = np.empty(n_atoms, dtype=np.float32)
    cov = np.empty(n_atoms, dtype=np.float32)
    elem = np.empty(n_atoms, dtype=np.int32)
    is_h = np.empty(n_atoms, dtype=bool)
    res_id = np.empty(n_atoms, dtype=np.int32)
    props = np.empty(n_atoms, dtype=np.int32)
    flags = np.empty(n_atoms, dtype=np.int32)
    occ = np.empty(n_atoms, dtype=np.float32)
    bval = np.empty(n_atoms, dtype=np.float32)
    alt_blank = np.empty(n_atoms, dtype=bool)
    alt_code = np.empty(n_atoms, dtype=np.uint8)
    model_arr = np.empty(n_atoms, dtype=np.int32)
    dot_idx = np.empty(n_atoms, dtype=np.int32)
    h_allow = np.zeros((n_atoms, 2), dtype=np.uint64)
    hv_bit = np.zeros((n_atoms, 2), dtype=np.uint64)
    atom_strs = []

    for i, a in enumerate(atoms):
        xyz[i, 0] = a["x"]
        xyz[i, 1] = a["y"]
        xyz[i, 2] = a["z"]
        rad[i] = a["radius"]
        cov[i] = a["covRad"]
        e = a["elem"]
        elem[i] = e
        ih = bool(atno_arr[e] == 1)
        is_h[i] = ih
        res_id[i] = a["res_obj_id"]
        props[i] = a["props"]
        flags[i] = a["flags"]
        occ[i] = a["occ"]
        bval[i] = a["bval"]
        ac = a["altloc"]
        alt_blank[i] = ac == " "
        alt_code[i] = ord(ac)
        model_arr[i] = a["model"]
        dot_idx[i] = co_dot_idx if (e == atom_id["atomC"] and a["is_co"]) else e
        if ih:
            bt = a["bondedto"]
            w0, w1 = all_heavy_bits if bt is None else bondedto_str_to_bits[bt]
            h_allow[i, 0] = w0
            h_allow[i, 1] = w1
        else:
            w0, w1 = heavy_name_to_bits.get(a["atomname"], default_heavy_bits)
            hv_bit[i, 0] = w0
            hv_bit[i, 1] = w1
        atom_strs.append(
            f"{a['chain']}{a['hy36']:4.4s}{a['icode']}{a['resname']} {a['atomname']}{ac}"
        )

    # Build touching neighbor lists via cKDTree.query_pairs
    max_touch = float(2.0 * np.max(rad) + 0.501)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=max_touch, output_type="ndarray")
    if len(pairs) == 0:
        return []

    u = pairs[:, 0]
    v = pairs[:, 1]
    same_mod = model_arr[u] == model_arr[v]
    comp_alt = (alt_code[u] == alt_code[v]) | alt_blank[u] | alt_blank[v]
    mask_pair = same_mod & comp_alt
    u = u[mask_pair]
    v = v[mask_pair]

    dvec = xyz[u] - xyz[v]
    dsq = np.sum(dvec * dvec, axis=-1).astype(np.float32)
    lim = rad[u] + rad[v] + np.float32(0.5)
    touch_ok = (dsq > np.float32(0.00001)) & (dsq <= (lim * lim))
    u = u[touch_ok]
    v = v[touch_ok]
    if len(u) == 0:
        return []

    # Directed edges (src -> nb) for valid SET1 source atoms
    src_ok_mask = ((flags & 1) != 0) & (elem != ignore_atom) & (occ >= OCCUPANCY_CUTOFF)
    dir_src = np.concatenate([u, v])
    dir_nb = np.concatenate([v, u])
    keep_dir = src_ok_mask[dir_src]
    dir_src = dir_src[keep_dir]
    dir_nb = dir_nb[keep_dir]
    if len(dir_src) == 0:
        return []

    # Sort by (dir_src, dir_nb) and group into CSR
    order = np.lexsort((dir_nb, dir_src))
    dir_src = dir_src[order]
    dir_nb = dir_nb[order]

    unique_src, starts, counts = np.unique(dir_src, return_index=True, return_counts=True)
    # Filter to unique_src having at least one SET2 neighbor (oktargsA in findTouchingAtoms)
    nb_is_set2 = ((flags[dir_nb] & 2) != 0).astype(np.int32)
    set2_counts = np.add.reduceat(nb_is_set2, starts)
    valid_src_mask = set2_counts > 0
    unique_src = unique_src[valid_src_mask]
    starts = starts[valid_src_mask]
    counts = counts[valid_src_mask]
    if len(unique_src) == 0:
        return []

    hb_s, hb_t, bo_s, bo_t, bo_g = call_gpu(
        "probe_dots",
        xyz,
        rad,
        cov,
        elem,
        is_h,
        res_id,
        props,
        flags,
        occ,
        alt_blank,
        h_allow,
        hv_bit,
        dot_idx,
        unique_src,
        starts,
        counts,
        dir_nb,
        batch_size,
    )

    hbond_keys = set()
    clash_best = {}  # unordered key -> (overlap_val, src_idx, nb_idx)

    for s_atom, t_atom in zip(hb_s, hb_t):
        s_str = atom_strs[s_atom]
        t_str = atom_strs[t_atom]
        key = (s_str + t_str) if s_str < t_str else (t_str + s_str)
        hbond_keys.add(key)

    for s_atom, t_atom, raw_gap in zip(bo_s, bo_t, bo_g):
        # writeRaw formats node->gap as "%.3f"
        overlap_val = float(f"{float(raw_gap):.3f}")
        if overlap_val <= -0.4:
            s_str = atom_strs[s_atom]
            t_str = atom_strs[t_atom]
            key = (s_str + t_str) if s_str < t_str else (t_str + s_str)
            prev = clash_best.get(key)
            if prev is None or overlap_val < prev[0]:
                clash_best[key] = (overlap_val, int(s_atom), int(t_atom))

    # Filter out H-bonded pairs and construct ProbeClash objects
    results = []
    for key, (overlap_val, s_atom, t_atom) in clash_best.items():
        if key in hbond_keys:
            continue
        s_str = atom_strs[s_atom]
        t_str = atom_strs[t_atom]
        a1 = decode_atom_string(s_str, use_segids, model_id)
        a2 = decode_atom_string(t_str, use_segids, model_id)
        atoms_pair = [a1, a2] if s_str < t_str else [a2, a1]
        results.append(
            ProbeClash(
                atoms_info=atoms_pair,
                overlap=overlap_val,
                probe_type="bo",
                max_b_factor=float(max(bval[s_atom], bval[t_atom])),
            )
        )
    return results


def count_probe_atoms(pdb_path, occupancy_max=10.0, owned_fn=None):
    """
    Reproduce `probe -q -mc -het -dumpatominfo "ogt%d not water"` on `core.pdb`
    without spawning a subprocess.
    """
    tbl = load_cpu_probe_tables()
    water_list = tbl["meta"]["WaterList"]
    ogt = 10
    if occupancy_max < ogt:
        ogt = int(occupancy_max)
    occ_thresh = ogt / 100.0

    count = 0
    with open(pdb_path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")[:80]
            if len(line) < 47:
                continue
            rec4 = line[:4].upper()
            if rec4 not in ("ATOM", "HETA"):
                continue
            if line[13].upper() == "Q":
                continue
            if owned_fn is not None and not owned_fn(line[20:22], line[22:26], line[26:27]):
                continue
            resname = line[17:20].upper()
            if resname in water_list:
                continue
            x = float(np.float32(float(line[30:38].strip() or 0.0)))
            y = float(np.float32(float(line[38:46].strip() or 0.0)))
            z = float(np.float32(float(line[46:54].strip() or 0.0)))
            if x >= 9998.0 and y >= 9998.0 and z >= 9998.0:
                continue
            occ = float(np.float32(float(line[54:60].strip()) if len(line) >= 55 and line[54:60].strip() else 1.0))
            if occ > occ_thresh:
                count += 1
    return count


def score_from_withh(
    withh_pdb,
    names,
    resnames,
    owned=lambda c, r: True,
    occupancy_max=10.0,
    use_segids=False,
    model_id="",
    core_tree=None,
    margin=25.0,
):
    """
    Compute block clashes and probe_atoms count from a protonated `withh_pdb` file
    on the GPU, matching `remolprobity.clashes.block_clashes`.
    """
    def model_key(chain, resseq, icode):
        c, r, _ = names[(chain.strip(), resseq.strip(), icode.strip())]
        return c, r

    def named(a):
        c, r, _ = names[(a.chain_id.strip(), a.resseq.strip(), a.icode.strip())]
        n = a.resname.strip()
        return (c, r, resnames.get(n, n), a.name.strip(), a.altloc.strip())

    def atom_key(a):
        return (
            a.chain_id.strip(),
            a.resseq.strip(),
            a.icode.strip(),
            a.resname.strip(),
            a.name.strip(),
            a.altloc.strip(),
        )

    xyz = _coordinates(withh_pdb)
    atoms = _atoms_by_residue(xyz)
    limit = margin - REACH

    def farthest(points):
        if core_tree is None:
            return None
        d, _ = core_tree.query(np.asarray(points))
        return float(np.max(d))

    def too_far(points):
        far = farthest(points)
        if far is None:
            return True, "the block owns nothing near it", None
        if far > limit:
            needed = math.ceil(far + REACH)
            return True, "use --margin %d or more" % needed, needed
        return False, None, None

    model_residue = {(c, residue_id(r, i)): v[:2] for (c, r, i), v in names.items()}
    check_cliques(
        _cliques(withh_pdb),
        lambda m: moved_points(atoms, m),
        core_tree,
        margin,
        lambda m: group_label(m, model_residue),
    )

    found = probe_clashes_jax(
        withh_pdb,
        occupancy_max=occupancy_max,
        use_segids=use_segids,
        model_id=model_id,
    )
    seen, records, worst = set(), [], None
    for c in sorted(found):
        a1, a2 = c.atoms_info
        key = c.id_str()
        if key in seen or not owned(*model_key(a1.chain_id, a1.resseq, a1.icode)):
            continue
        seen.add(key)
        p1, p2 = xyz[atom_key(a1)], xyz[atom_key(a2)]
        if core_tree is not None or margin is not None:
            if core_tree is not None:
                bad, advice, needed = too_far([p1, p2])
                if bad:
                    message = "the clash %s - %s reaches within %.0f Å of the margin; %s" % (
                        named(a1),
                        named(a2),
                        REACH,
                        advice,
                    )
                    if needed is None:
                        raise MarginError(message, needed)
                    if worst is None or needed > worst[0]:
                        worst = (needed, message)
        records.append((named(a1), named(a2), abs(c.overlap), math.dist(p1, p2)))
    if worst is not None:
        raise MarginError(worst[1], worst[0])

    n_atoms = count_probe_atoms(
        withh_pdb,
        occupancy_max=occupancy_max,
        owned_fn=lambda c, r, i: owned(*model_key(c, r, i)),
    )
    return {"records": records, "probe_atoms": n_atoms}


def block_clashes(noh, conv, owned, core_tree, margin, tools, workdir, occupancy_max, stretch=None):
    """
    Drop-in GPU replacement for `remolprobity.clashes.block_clashes`.
    Runs `reduce` (`add_hydrogens`) to place hydrogens and then evaluates clashes
    and `probe_atoms` on the GPU via `score_from_withh`.
    """
    from mmtbx.validation import utils

    h, names = pdb_copy(noh, conv)
    resnames = {}
    for ag_model, ag_pdb in zip(noh.atom_groups(), h.atom_groups()):
        resnames[ag_pdb.resname.strip()] = ag_model.resname.strip()

    noh_pdb = os.path.join(workdir, "noh.pdb")
    restore = {}
    text = h.as_pdb_string()
    if stretch is not None:
        renamed, restore = spare_chain_ids(h, lambda c, r, i: stretch[names[(c, r, i)][:2]])
        text = rename_chain_ids(text, renamed)
    with open(noh_pdb, "w") as f:
        f.write(text)
    use_segids = utils.use_segids_in_place_of_chainids(hierarchy=h)
    model_id = h.models()[0].id
    del h
    withh = add_hydrogens(tools, noh_pdb, workdir)
    os.remove(noh_pdb)
    restore_chain_ids(withh, restore)

    try:
        return score_from_withh(
            withh,
            names=names,
            resnames=resnames,
            owned=owned,
            occupancy_max=occupancy_max,
            use_segids=use_segids,
            model_id=model_id,
            core_tree=core_tree,
            margin=margin,
        )
    finally:
        if os.path.exists(withh):
            os.remove(withh)
