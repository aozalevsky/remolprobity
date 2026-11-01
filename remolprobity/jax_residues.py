"""
JAX/GPU implementation of Ramachandran (ramalyze) and Rotamer (rotalyze)
validation, bit-exact with stock mmtbx.validation.ramalyze and rotalyze.
"""
import os
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .ids import residue_id
from .jax_io import JaxChain, JaxHierarchy, JaxModel, prepare_noh_hierarchy, read_hierarchy, _hy36decode
from .jax_tables import (
    COMMON_AMINO_ACIDS,
    RAMA_CISPRO,
    RAMA_GENERAL,
    RAMA_GLYCINE,
    RAMA_ILEVAL,
    RAMA_PREPRO,
    RAMA_TRANSPRO,
    RAMA_TYPE_NAMES,
    SYM_AA,
    get_cablam_tables,
    get_tables,
)


@jax.jit
def dihedral_angles_jax(p0, p1, p2, p3):
    """
    Vectorized dihedral angle in degrees matching scitbx::math::dihedral_angle
    (scitbx/math/dihedral.h:40-63).
    Inputs: (N, 3) float64 arrays.
    Output: (N,) float64 array in [-180, 180].
    """
    d_01 = p0 - p1
    d_21 = p2 - p1
    d_23 = p2 - p3
    n_0121 = jnp.cross(d_01, d_21)
    n_2123 = jnp.cross(d_21, d_23)
    n_0121_norm = jnp.sum(n_0121 * n_0121, axis=-1)
    n_2123_norm = jnp.sum(n_2123 * n_2123, axis=-1)
    denom = jnp.sqrt(n_0121_norm * n_2123_norm)
    safe_denom = jnp.where(denom == 0.0, 1.0, denom)
    cos_angle = jnp.clip(jnp.sum(n_0121 * n_2123, axis=-1) / safe_denom, -1.0, 1.0)
    angle = jnp.arccos(cos_angle) * (180.0 / jnp.pi)
    cross_normals = jnp.cross(n_0121, n_2123)
    sign_dot = jnp.sum(d_21 * cross_normals, axis=-1)
    angle = jnp.where(sign_dot < 0.0, -angle, angle)
    return angle


def _wrap_angle_180(val):
    """Wrap angle into [-180, 180] matching rama_eval::bring_angle_to_range."""
    # For angles already in [-180, 180] from dihedral_angle, this is identity,
    # but handles out-of-range values identically to the C++ while loop.
    v = jnp.where(val > 180.0, val - 360.0 * jnp.ceil((val - 180.0) / 360.0), val)
    v = jnp.where(v < -180.0, v + 360.0 * jnp.ceil((-180.0 - v) / 360.0), v)
    return v


def _rama_bin_and_bounds(val):
    """
    Exact replica of rama_eval::get_bin_and_bounds (rama_eval.h:35-51).
    Bins have centers at -179, -177, ..., 179 (width 2.0).
    """
    lower_val = jnp.floor(val)
    # In C++: if ((int)lower_val % 2 == 0) lower_val -= 1;
    # Note: C++ `% 2 == 0` is true for both positive and negative even integers.
    is_even = (jnp.trunc(lower_val).astype(jnp.int64) % 2) == 0
    lower_val = jnp.where(is_even, lower_val - 1.0, lower_val)
    higher_val = lower_val + 2.0
    lower_bin = jnp.trunc((lower_val + 179.0) / 2.0).astype(jnp.int64) % 180
    lower_bin = jnp.where(lower_bin < 0, lower_bin + 180, lower_bin)
    higher_bin = (lower_bin + 1) % 180
    higher_bin = jnp.where(higher_bin < 0, higher_bin + 180, higher_bin)
    return lower_bin, higher_bin, lower_val, higher_val


def _linear_interp_1d(x1, x2, y1, y2, x):
    """
    Exact replica of scitbx::math::linear_interpolation (linear_interpolation.h:26-41).
    """
    # When x1 < x2 (always true since higher_val = lower_val + 2):
    # a = (y1 - y2) / (x1 - x2); b = y1 - a * x1; res = a * x + b
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    return a * x + b


@jax.jit
def rama_eval_batch_jax(
    ca0, c0, n1, ca1, c1, n2, omega_ca0, omega_c0, omega_n1, omega_ca1,
    is_gly, is_pro, is_ile_val, next_is_pro, rama_tables
):
    """
    Vectorized GPU kernel for Ramachandran angles, residue type (cis vs trans Pro),
    2D bilinear table interpolation, score, and classification.
    """
    phi = dihedral_angles_jax(c0, n1, ca1, c1)
    psi = dihedral_angles_jax(n1, ca1, c1, n2)
    omega = dihedral_angles_jax(omega_ca0, omega_c0, omega_n1, omega_ca1)
    is_cis = (omega > -90.0) & (omega < 90.0)

    rama_type = jnp.full(phi.shape, RAMA_GENERAL, dtype=jnp.int32)
    rama_type = jnp.where(is_ile_val, RAMA_ILEVAL, rama_type)
    rama_type = jnp.where(next_is_pro, RAMA_PREPRO, rama_type)
    rama_type = jnp.where(is_pro & (~is_cis), RAMA_TRANSPRO, rama_type)
    rama_type = jnp.where(is_pro & is_cis, RAMA_CISPRO, rama_type)
    rama_type = jnp.where(is_gly, RAMA_GLYCINE, rama_type)

    phi_w = _wrap_angle_180(phi)
    psi_w = _wrap_angle_180(psi)
    lb_phi, hb_phi, lv_phi, hv_phi = _rama_bin_and_bounds(phi_w)
    lb_psi, hb_psi, lv_psi, hv_psi = _rama_bin_and_bounds(psi_w)

    v1 = rama_tables[rama_type, lb_phi, lb_psi]
    v2 = rama_tables[rama_type, hb_phi, lb_psi]
    v3 = rama_tables[rama_type, hb_phi, hb_psi]
    v4 = rama_tables[rama_type, lb_phi, hb_psi]

    # scitbx::math::linear_interpolation_2d(lv_phi, lv_psi, hv_phi, hv_psi, v1, v2, v3, v4, phi_w, psi_w)
    v14 = _linear_interp_1d(lv_phi, hv_phi, v1, v2, phi_w)
    v32 = _linear_interp_1d(lv_phi, hv_phi, v4, v3, phi_w)
    score_raw = _linear_interp_1d(lv_psi, hv_psi, v14, v32, psi_w)
    score = score_raw * 100.0

    # Classification: 0=OUTLIER, 1=Allowed, 2=Favored
    allowed_thresh = jnp.where(
        rama_type == RAMA_GENERAL,
        0.0005,
        jnp.where(rama_type == RAMA_CISPRO, 0.0020, 0.0010),
    )
    eval_code = jnp.where(
        score_raw >= 0.02,
        2,
        jnp.where(score_raw >= allowed_thresh, 1, 0),
    )
    return phi, psi, score, rama_type, eval_code


@jax.jit
def rotamer_eval_batch_jax(
    p0, p1, p2, p3, chi_valid, aa_idx, is_sym,
    rota_flat, rota_offset, rota_ndim, rota_min, rota_max, rota_nbins, rota_dowrap, rota_wbin,
    box_low, box_high, box_valid
):
    """
    Vectorized GPU kernel for sidechain chi1..4 dihedral calculation,
    4D NDimTable multilinear interpolation, evaluation threshold, and
    RotamerID box classification.
    p0, p1, p2, p3: (N, 4, 3) float64 arrays of atom coordinates for chi1..4.
    chi_valid: (N, 4) bool array.
    aa_idx: (N,) int32 index into aa_list (0..18).
    is_sym: (N,) bool (True for asp, glu, phe, tyr).
    """
    # Compute all 4 chi angles in parallel: (N, 4)
    chis = dihedral_angles_jax(p0, p1, p2, p3)
    chis = jnp.where(chi_valid, chis, 0.0)

    # NDimTable.whereIs(pt) and valueAt(pt) (n_dim_table.py:71-196)
    min_v = rota_min[aa_idx]        # (N, 4)
    n_b = rota_nbins[aa_idx]        # (N, 4) int32
    do_w = rota_dowrap[aa_idx] == 1 # (N, 4) bool
    w_b = rota_wbin[aa_idx]         # (N, 4)
    ndim = rota_ndim[aa_idx]        # (N,) int32
    base_off = rota_offset[aa_idx]  # (N,) int64

    dim_mask = jnp.arange(4, dtype=jnp.int32)[None, :] < ndim[:, None]  # (N, 4)

    # whereIs: int(min(floor((coords[i] - minVal[i]) / wBin[i]), nBins[i] - 1))
    n_b_f = n_b.astype(jnp.float64)
    va_home = jnp.minimum(jnp.floor((chis - min_v) / w_b), n_b_f - 1.0).astype(jnp.int32)
    va_home = jnp.where(dim_mask, va_home, 0)

    # centerOf: minVal[i] + wBin[i] * (va_home[i] + 0.5)
    va_home_ctr = min_v + w_b * (va_home.astype(jnp.float64) + 0.5)

    # va_neighbor and va_contrib (n_dim_table.py:173-175)
    va_neighbor = jnp.where(chis < va_home_ctr, va_home - 1, va_home + 1)
    va_neighbor = jnp.where(dim_mask, va_neighbor, 0)
    va_contrib = jnp.abs((chis - va_home_ctr) / w_b)
    va_contrib = jnp.where(dim_mask, va_contrib, 0.0)

    # Helper for bin2index_limit per dimension (wrapbin then clampInclusive(0, iBin, nBins - 1))
    def _limit_bin(b_val, d):
        nb_d = n_b[:, d]
        wrapped = jnp.mod(b_val, nb_d)
        wrapped = jnp.where(wrapped < 0, wrapped + nb_d, wrapped)
        b_wrapped = jnp.where(do_w[:, d], wrapped, b_val)
        return jnp.clip(b_wrapped, 0, nb_d - 1).astype(jnp.int64)

    home_lim = [_limit_bin(va_home[:, d], d) for d in range(4)]
    neigh_lim = [_limit_bin(va_neighbor[:, d], d) for d in range(4)]

    # 16-corner 4D multilinear interpolation matching NDimTable.valueAt loop order
    result = jnp.zeros(chis.shape[0], dtype=jnp.float64)
    for b_idx in range(16):
        coeff = jnp.ones(chis.shape[0], dtype=jnp.float64)
        lin_idx = jnp.zeros(chis.shape[0], dtype=jnp.int64)
        for d in range(4):
            active = d < ndim
            bit_zero = (b_idx & (1 << d)) == 0
            c_d = jnp.where(bit_zero, 1.0 - va_contrib[:, d], va_contrib[:, d])
            coeff = jnp.where(active, coeff * c_d, coeff)
            idx_d = jnp.where(bit_zero, home_lim[d], neigh_lim[d])
            nb_d = n_b[:, d].astype(jnp.int64)
            lin_idx = jnp.where(active, lin_idx * nb_d + idx_d, lin_idx)
        valid_corner = b_idx < (1 << ndim)
        val_corner = rota_flat[base_off + lin_idx].astype(jnp.float64)
        result = jnp.where(valid_corner, result + coeff * val_corner, result)

    score = result * 100.0
    # Evaluation: 0=OUTLIER, 1=Allowed, 2=Favored (PCT_AA_FAVORED=0.02, PCT_AA_ALLOWED=0.003)
    eval_code = jnp.where(result >= 0.02, 2, jnp.where(result >= 0.003, 1, 0))

    # RotamerID.wrap_chis and wrap_sym (rotamer_eval.py:345-419)
    # Note: chis has n_chis elements (for PRO, 3 chis; for others, ndim chis).
    # However, PRO has 3 box bounds in rotamer_names.props or 1? Let's check:
    # box_valid / box_low / box_high only constrain the dimensions present in bounds!
    w_chis = jnp.mod(chis, 360.0)
    w_chis = jnp.where(w_chis < 0.0, w_chis + 360.0, w_chis)
    # Sym AA (asp, glu, phe, tyr): wrap last chi (at index ndim - 1) mod 180
    last_idx = ndim - 1
    is_last = jnp.arange(4, dtype=jnp.int32)[None, :] == last_idx[:, None]
    sym_wrapped = jnp.mod(w_chis, 180.0)
    sym_wrapped = jnp.where(sym_wrapped < 0.0, sym_wrapped + 180.0, sym_wrapped)
    w_chis = jnp.where(is_sym[:, None] & is_last, sym_wrapped, w_chis)

    # Match against rotamer boxes for aa_idx: box_low[aa_idx] is (N, max_boxes, 4)
    lows = box_low[aa_idx]      # (N, B, 4)
    highs = box_high[aa_idx]    # (N, B, 4)
    valids = box_valid[aa_idx]  # (N, B)
    in_dim = (w_chis[:, None, :] >= lows) & (w_chis[:, None, :] <= highs)
    in_dim = jnp.where(chi_valid[:, None, :], in_dim, True)
    in_box = valids & jnp.all(in_dim, axis=-1)  # (N, B)
    any_match = jnp.any(in_box, axis=-1)
    first_box = jnp.argmax(in_box, axis=-1).astype(jnp.int32)
    box_idx = jnp.where(any_match, first_box, -1)

    return score, eval_code, box_idx


def _select_backbone_chain(chain):
    """
    Create a copy of `chain` containing only backbone atoms {"N", "CA", "C", "O", "CB", "H", "HNO"},
    matching mmtbx.validation.utils.get_atom_selection("name N or name CA or name C or name O or name CB or name H or name HNO")
    prior to building conformers in generate_protein_threes.
    """
    bb_names = {"N", "CA", "C", "O", "CB", "H", "HNO"}
    new_ch = JaxChain(chain.id)
    for rg in chain.residue_groups():
        new_rg = None
        for ag in rg.atom_groups():
            new_ag = None
            for a in ag.atoms():
                if a.name.strip() in bb_names:
                    if new_rg is None:
                        from .jax_io import JaxResidueGroup
                        new_rg = JaxResidueGroup(rg.resseq, rg.icode, rg.link_to_previous)
                    if new_ag is None:
                        from .jax_io import JaxAtomGroup
                        new_ag = JaxAtomGroup(ag.altloc, ag.resname)
                    new_ag.append_atom(a)
            if new_ag is not None:
                new_rg.append_atom_group(new_ag)
        if new_rg is not None:
            new_ch.append_residue_group(new_rg)
    return new_ch


def _get_altloc_from_three(r0, r1, r2, conf_altloc, r1_is_pure_main_conf):
    """
    Exact replica of ramalyze.get_altloc_from_three(three) (ramalyze.py:660-680)
    combined with three.get_dummy_dihedral_atom (multi_residue_class.py:92-116).
    """
    if r1_is_pure_main_conf:
        for a in r1.atoms():
            ag = a.parent()
            al = ag.altloc if ag is not None else ""
            if al != "":
                return al
        c0 = r0.find_atom_by(" C  ")
        if c0 is not None:
            ag = c0.parent()
            al = ag.altloc if ag is not None else ""
            if al != "":
                return al
        n2 = r2.find_atom_by(" N  ")
        if n2 is not None:
            ag = n2.parent()
            al = ag.altloc if ag is not None else ""
            if al != "":
                return al
        return ""
    return conf_altloc


def _next_bucket(n):
    b = 64
    while b < n:
        b <<= 2 if b < 16384 else 1
    return b


def _pad_to(arr, target_n):
    n = arr.shape[0]
    if n == target_n:
        return arr
    pad_width = [(0, target_n - n)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_width, mode="edge")


def _ensure_hierarchy(h_or_path):
    if isinstance(h_or_path, str):
        return prepare_noh_hierarchy(read_hierarchy(h_or_path), in_place=True)
    return h_or_path


_SLOT_NAMES = (
    "N", "CA", "C", "O", "CB",
    "CD", "CD1", "CE", "CG", "CG1", "CZ",
    "ND1", "NE", "NZ", "OD1", "OE1", "OG", "OG1", "SD", "SE", "SG",
)
_ATOM_SLOT = {}
for _si, _sn in enumerate(_SLOT_NAMES):
    _ATOM_SLOT[_sn] = _si
    _ATOM_SLOT[f" {_sn:<3s}"] = _si
    _ATOM_SLOT[f"{_sn:<4s}"] = _si

_BB_EXTRA_NAMES = frozenset(("H", " H  ", "H   ", "HNO", " HNO", "HNO "))
_NA_SINGLE_ATOM_VALID = frozenset((
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    " P  ", " OP1", " OP2", " O5'", " C5'", " C4'", " O4'", " C3'", " O3'", " C2'", " O2'", " C1'",
))
_NA_SINGLE_RESNAMES = frozenset(("A", "C", "G", "U", "DA", "DC", "DG", "DT", "T"))
_STATIC_META = None


def _get_static_meta():
    global _STATIC_META
    if _STATIC_META is not None:
        return _STATIC_META
    tables = get_tables()
    cab_tables = get_cablam_tables()
    aa_list = list(tables["aa_list"])
    aa_to_idx = tables["aa_to_idx"]
    chi_atoms_map = tables["chi_atoms"]

    def_names = aa_list + ["mse"]
    def_to_idx = {rn: idx for idx, rn in enumerate(def_names)}
    chi_slots = np.zeros((len(def_names), 4, 4), dtype=np.int32)
    chi_req = np.zeros((len(def_names), 4), dtype=bool)
    for idx, rn in enumerate(def_names):
        c_defs = chi_atoms_map[rn]
        for ci, c_atoms in enumerate(c_defs):
            chi_req[idx, ci] = True
            for ai, an in enumerate(c_atoms):
                chi_slots[idx, ci, ai] = _ATOM_SLOT[an.strip()]

    unit_pts = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    _STATIC_META = {
        "def_to_idx": def_to_idx,
        "aa_to_idx": aa_to_idx,
        "chi_slots": chi_slots,
        "chi_req": chi_req,
        "unit_pts": unit_pts,
        "common_aa": cab_tables["common_aa"],
        "all_aa": cab_tables["all_aa"],
        "all_na": cab_tables["all_na"],
        "d_aa": cab_tables["d_aa"],
        "d_to_l": cab_tables["d_to_l"],
    }
    return _STATIC_META


def _ensure_flat_index(h):
    """
    Build and cache a columnar index on JaxHierarchy `h` in a single pass so
    ramachandran, rotamers, cbeta, omegas, and cablam can execute via vectorized
    NumPy/JAX array indexing without repeating Python conformer/atom traversals.
    """
    if not isinstance(h, JaxHierarchy):
        return None
    if getattr(h, "_flat_index", None) is not None:
        return h._flat_index

    smeta = _get_static_meta()
    def_to_idx = smeta["def_to_idx"]
    aa_to_idx = smeta["aa_to_idx"]
    common_aa = smeta["common_aa"]
    all_aa = smeta["all_aa"]
    all_na = smeta["all_na"]
    d_aa = smeta["d_aa"]
    d_to_l = smeta["d_to_l"]
    atom_slot_get = _ATOM_SLOT.get

    atoms = h.atoms()
    if not atoms:
        h._flat_index = {"xyz": np.zeros((0, 3), dtype=np.float64), "occ": np.zeros(0, dtype=np.float64), "chains": {}}
        return h._flat_index

    xyz = np.array([a.xyz for a in atoms], dtype=np.float64)
    occ = np.array([a.occ for a in atoms], dtype=np.float64)

    resid_cache = {}
    resname_cache = {}
    chains_map = {}

    model = h.models()[0]
    for ch in model.chains():
        rgs = ch._residue_groups
        m = len(rgs)
        if m == 0:
            continue
        is_pure = True
        seen_rc = set()
        for rg in rgs:
            ags = rg._atom_groups
            if len(ags) != 1 or ags[0].altloc != "":
                is_pure = False
                break
            rc_key = (rg.resseq, rg.icode)
            if rc_key in seen_rc:
                is_pure = False
                break
            seen_rc.add(rc_key)
        if not is_pure:
            chains_map[id(ch)] = None
            continue

        cid = ch.id
        cid_strip = cid.strip()
        slot_rows = []
        has_bb_list = []
        resnames = []
        resids = []
        resseqs = []
        icodes = []
        resseq_ints = []
        flags_list = []
        rota_def_list = []
        rota_aa_list = []
        cbeta_rt_list = []
        cab_tidx_list = []
        cab_classes = []

        prev_ri = -1 << 60
        prev_ic = ""
        is_sorted = True

        for rg in rgs:
            rc_key = (rg.resseq, rg.icode)
            rc_val = resid_cache.get(rc_key)
            if rc_val is None:
                rc_val = (residue_id(rg.resseq, rg.icode), _hy36decode(rg.resseq))
                resid_cache[rc_key] = rc_val
            resid_str, ri = rc_val
            ic = rg.icode
            if ri < prev_ri or (ri == prev_ri and ic < prev_ic):
                is_sorted = False
            prev_ri = ri
            prev_ic = ic

            ag = rg._atom_groups[0]
            raw_rn = ag.resname
            rn_info = resname_cache.get(raw_rn)
            if rn_info is None:
                rn_s = raw_rn.strip()
                rn_u = rn_s.upper()
                rn_l = rn_s.lower()
                is_c_aa = rn_s in COMMON_AMINO_ACIDS
                is_a_aa = rn_s in all_aa
                is_d = rn_s in d_aa
                is_g = rn_s == "GLY"
                is_p = rn_s == "PRO"
                is_iv = rn_s in ("ILE", "VAL")
                is_sym = rn_l in SYM_AA
                fl = (
                    (1 if is_c_aa else 0)
                    | (2 if is_a_aa else 0)
                    | (4 if is_d else 0)
                    | (8 if is_g else 0)
                    | (16 if is_p else 0)
                    | (32 if is_iv else 0)
                    | (64 if is_sym else 0)
                )
                if is_c_aa and rn_l not in ("ala", "gly"):
                    eval_r = "met" if rn_l == "mse" else rn_l
                    r_aa = aa_to_idx.get(eval_r, -1)
                    r_def = def_to_idx.get(rn_l, -1) if r_aa >= 0 else -1
                else:
                    r_aa = -1
                    r_def = -1

                if is_g:
                    cb_rt = -1
                elif rn_s in ("ALA", "DAL"):
                    cb_rt = 1
                elif rn_s in ("PRO", "DPR"):
                    cb_rt = 2
                elif rn_s in ("VAL", "THR", "ILE", "DVA", "DTH", "DIL"):
                    cb_rt = 3
                else:
                    cb_rt = 0

                if rn_u in common_aa:
                    cab_t = 1 if rn_u == "GLY" else (2 if rn_u == "PRO" else 0)
                else:
                    cab_t = -1

                cab_rn = d_to_l.get(rn_u, rn_u) if rn_u else None
                not_hoh = int(cab_rn != "HOH")
                cab_aa = int(cab_rn in all_aa) if cab_rn is not None else 0
                cab_na = int(cab_rn in all_na) if cab_rn is not None else 0
                cab_unk = int(cab_rn == "UNK")
                chk_na_1 = cab_rn in _NA_SINGLE_RESNAMES

                rn_info = (rn_s, fl, r_def, r_aa, cb_rt, cab_t, not_hoh, cab_aa, cab_na, cab_unk, chk_na_1)
                resname_cache[raw_rn] = rn_info

            rn_s, fl, r_def, r_aa, cb_rt, cab_t, not_hoh, cab_aa, cab_na, cab_unk, chk_na_1 = rn_info
            ag_atoms = ag._atoms
            if chk_na_1 and len(ag_atoms) == 1 and ag_atoms[0].name not in _NA_SINGLE_ATOM_VALID:
                if ag_atoms[0].name.strip() not in _NA_SINGLE_ATOM_VALID:
                    not_hoh, cab_aa, cab_na, cab_unk = 1, 0, 0, 0

            row = [-1] * 21
            has_bb = False
            for a in ag_atoms:
                an = a.name
                s = atom_slot_get(an)
                if s is None:
                    s = atom_slot_get(an.strip())
                if s is not None:
                    row[s] = a.i_seq
                    if s <= 4:
                        has_bb = True
                elif an in _BB_EXTRA_NAMES or an.strip() in _BB_EXTRA_NAMES:
                    has_bb = True

            slot_rows.append(row)
            has_bb_list.append(has_bb)
            resnames.append(rn_s)
            resids.append(resid_str)
            resseqs.append(rg.resseq)
            icodes.append(ic)
            resseq_ints.append(ri)
            flags_list.append(fl)
            rota_def_list.append(r_def)
            rota_aa_list.append(r_aa)
            cbeta_rt_list.append(cb_rt)
            cab_tidx_list.append(cab_t)
            cab_classes.append((cid_strip, resid_str, not_hoh, cab_aa, cab_na, cab_unk))

        chains_map[id(ch)] = {
            "cid": cid,
            "cid_strip": cid_strip,
            "m": m,
            "is_sorted": is_sorted,
            "slot_idx": np.array(slot_rows, dtype=np.int32),
            "has_bb": np.array(has_bb_list, dtype=bool),
            "resnames": resnames,
            "resids": resids,
            "resseqs": resseqs,
            "icodes": icodes,
            "resseq_ints": np.array(resseq_ints, dtype=np.int64),
            "flags": np.array(flags_list, dtype=np.int32),
            "rota_def_idx": np.array(rota_def_list, dtype=np.int32),
            "rota_aa_idx": np.array(rota_aa_list, dtype=np.int32),
            "cbeta_rtype": np.array(cbeta_rt_list, dtype=np.int32),
            "cab_tidx": np.array(cab_tidx_list, dtype=np.int32),
            "cab_classes": cab_classes,
        }

    h._flat_index = {"xyz": xyz, "occ": occ, "chains": chains_map}
    return h._flat_index


def ramachandran(hierarchy):
    """
    Every ramalyze result as (chain, resid, resname, altloc, class, phi, psi, score).
    Accepts a JaxHierarchy, an iotbx.pdb.hierarchy.root, or a path to a PDB/mmCIF file.
    Evaluates all candidate peptide triples across the entire model in a single
    vectorized JAX/GPU batch while preserving exact per-chain deduplication and sort order.
    """
    h = _ensure_hierarchy(hierarchy)
    flat = _ensure_flat_index(h)
    xyz_all = flat["xyz"] if flat is not None else None
    chains_flat = flat["chains"] if flat is not None else {}

    tables = get_tables()
    model = h.models()[0]

    # Collect unique chain IDs in order of appearance, matching residues.ramachandran
    chain_ids_ordered = []
    chains_by_cid = {}
    for ch in model.chains():
        cid = ch.id
        if cid not in chains_by_cid:
            chain_ids_ordered.append(cid)
            chains_by_cid[cid] = []
        chains_by_cid[cid].append(ch)

    triples_meta = []
    ca0_chunks, c0_chunks, n1_chunks, ca1_chunks, c1_chunks, n2_chunks = [], [], [], [], [], []
    om_ca0_chunks, om_c0_chunks, om_n1_chunks, om_ca1_chunks = [], [], [], []
    is_gly_chunks, is_pro_chunks, is_ile_val_chunks, next_is_pro_chunks = [], [], [], []
    cid_all_pure_sorted = {}

    for cid in chain_ids_ordered:
        uniqueness_keys = set()
        ch_list = chains_by_cid[cid]
        can_skip_sort = len(ch_list) == 1
        for ch in ch_list:
            ce = chains_flat.get(id(ch))
            if ce is not None:
                if not ce["is_sorted"]:
                    can_skip_sort = False
                flags = ce["flags"]
                bb_aa = np.flatnonzero(((flags & 1) != 0) & ce["has_bb"])
                if len(bb_aa) < 3:
                    continue
                r0 = bb_aa[:-2]
                r1 = bb_aa[1:-1]
                r2 = bb_aa[2:]
                s_idx = ce["slot_idx"]
                c0_i = s_idx[r0, 2]
                n1_i = s_idx[r1, 0]
                ca1_i = s_idx[r1, 1]
                c1_i = s_idx[r1, 2]
                n2_i = s_idx[r2, 0]
                valid = (c0_i >= 0) & (n1_i >= 0) & (ca1_i >= 0) & (c1_i >= 0) & (n2_i >= 0)
                if not np.any(valid):
                    continue
                r0, r1, r2 = r0[valid], r1[valid], r2[valid]
                c0_i, n1_i, ca1_i, c1_i, n2_i = c0_i[valid], n1_i[valid], ca1_i[valid], c1_i[valid], n2_i[valid]

                d01 = np.sum((xyz_all[c0_i] - xyz_all[n1_i]) ** 2, axis=-1) < 4.0
                d12 = np.sum((xyz_all[c1_i] - xyz_all[n2_i]) ** 2, axis=-1) < 4.0
                valid2 = d01 & d12
                if not np.any(valid2):
                    continue
                r0, r1, r2 = r0[valid2], r1[valid2], r2[valid2]
                c0_i, n1_i, ca1_i, c1_i, n2_i = c0_i[valid2], n1_i[valid2], ca1_i[valid2], c1_i[valid2], n2_i[valid2]

                resseqs = ce["resseqs"]
                icodes = ce["icodes"]
                resids = ce["resids"]
                resnames = ce["resnames"]
                resseq_ints = ce["resseq_ints"]
                ch_id = ce["cid"]
                ch_strip = ce["cid_strip"]

                if len(ch_list) > 1 or uniqueness_keys:
                    keep = []
                    for idx_k, r1_k in enumerate(r1.tolist()):
                        rk = (ch_id, resseqs[r1_k], icodes[r1_k])
                        if rk not in uniqueness_keys:
                            uniqueness_keys.add(rk)
                            keep.append(idx_k)
                    if not keep:
                        continue
                    if len(keep) < len(r1):
                        keep_arr = np.array(keep, dtype=np.int64)
                        r0, r1, r2 = r0[keep_arr], r1[keep_arr], r2[keep_arr]
                        c0_i, n1_i, ca1_i, c1_i, n2_i = (
                            c0_i[keep_arr], n1_i[keep_arr], ca1_i[keep_arr], c1_i[keep_arr], n2_i[keep_arr]
                        )

                ca0_i = s_idx[r0, 1]
                has_om = ca0_i >= 0
                c0_xyz = xyz_all[c0_i]
                n1_xyz = xyz_all[n1_i]
                ca1_xyz = xyz_all[ca1_i]
                c1_xyz = xyz_all[c1_i]
                n2_xyz = xyz_all[n2_i]
                ca0_xyz = np.where(has_om[:, None], xyz_all[np.maximum(ca0_i, 0)], 0.0)

                c0_chunks.append(c0_xyz)
                n1_chunks.append(n1_xyz)
                ca1_chunks.append(ca1_xyz)
                c1_chunks.append(c1_xyz)
                n2_chunks.append(n2_xyz)
                ca0_chunks.append(ca0_xyz)
                om_ca0_chunks.append(ca0_xyz)
                if np.all(has_om):
                    om_c0_chunks.append(c0_xyz)
                    om_n1_chunks.append(n1_xyz)
                    om_ca1_chunks.append(ca1_xyz)
                else:
                    om_c0_chunks.append(np.where(has_om[:, None], c0_xyz, np.array([1.0, 0.0, 0.0])))
                    om_n1_chunks.append(np.where(has_om[:, None], n1_xyz, np.array([0.0, 1.0, 0.0])))
                    om_ca1_chunks.append(np.where(has_om[:, None], ca1_xyz, np.array([0.0, 0.0, 1.0])))

                fl1 = flags[r1]
                fl2 = flags[r2]
                is_gly_chunks.append((fl1 & 8) != 0)
                is_pro_chunks.append((fl1 & 16) != 0)
                is_ile_val_chunks.append((fl1 & 32) != 0)
                next_is_pro_chunks.append((fl2 & 16) != 0)

                r1_list = r1.tolist()
                ri_list = resseq_ints[r1].tolist()
                triples_meta.extend(
                    (cid, ch_id, ch_strip, resids[k], icodes[k], ri, resnames[k], "")
                    for k, ri in zip(r1_list, ri_list)
                )
            else:
                can_skip_sort = False
                ca0_l, c0_l, n1_l, ca1_l, c1_l, n2_l = [], [], [], [], [], []
                om_ca0_l, om_c0_l, om_n1_l, om_ca1_l = [], [], [], []
                is_gly_l, is_pro_l, is_ile_val_l, next_is_pro_l = [], [], [], []
                ch_strip = ch.id.strip()
                bb_ch = _select_backbone_chain(ch)
                for conf in bb_ch.conformers():
                    conf_alt = conf.altloc
                    three = []
                    for res in conf.residues():
                        rn = res.resname.strip()
                        if rn not in COMMON_AMINO_ACIDS:
                            continue
                        if three:
                            prev_c = three[-1].find_atom_by(" C  ")
                            curr_n = res.find_atom_by(" N  ")
                            if prev_c is None or curr_n is None:
                                three = [res]
                                continue
                            dx = prev_c.xyz[0] - curr_n.xyz[0]
                            dy = prev_c.xyz[1] - curr_n.xyz[1]
                            dz = prev_c.xyz[2] - curr_n.xyz[2]
                            if dx * dx + dy * dy + dz * dz >= 4.0:
                                three = [res]
                                continue
                        three.append(res)
                        if len(three) > 3:
                            three.pop(0)
                        if len(three) < 3:
                            continue

                        r0, r1, r2 = three[0], three[1], three[2]
                        c0 = r0.find_atom_by(" C  ")
                        n1 = r1.find_atom_by(" N  ")
                        ca1 = r1.find_atom_by(" CA ")
                        c1 = r1.find_atom_by(" C  ")
                        n2 = r2.find_atom_by(" N  ")
                        if c0 is None or n1 is None or ca1 is None or c1 is None or n2 is None:
                            continue

                        altloc = _get_altloc_from_three(r0, r1, r2, conf_alt, r1.is_pure_main_conf)
                        result_key = (ch.id, r1.resseq, r1.icode)
                        if altloc == "" and result_key in uniqueness_keys:
                            continue
                        uniqueness_keys.add(result_key)

                        rn1 = r1.resname.strip()
                        rn2 = r2.resname.strip()
                        ca0 = r0.find_atom_by(" CA ")
                        has_omega = ca0 is not None
                        ca0_xyz = ca0.xyz if has_omega else (0.0, 0.0, 0.0)

                        c0_l.append(c0.xyz)
                        n1_l.append(n1.xyz)
                        ca1_l.append(ca1.xyz)
                        c1_l.append(c1.xyz)
                        n2_l.append(n2.xyz)
                        ca0_l.append(ca0_xyz)
                        om_ca0_l.append(ca0_xyz)
                        om_c0_l.append(c0.xyz if has_omega else (1.0, 0.0, 0.0))
                        om_n1_l.append(n1.xyz if has_omega else (0.0, 1.0, 0.0))
                        om_ca1_l.append(ca1.xyz if has_omega else (0.0, 0.0, 1.0))

                        is_gly_l.append(rn1 == "GLY")
                        is_pro_l.append(rn1 == "PRO")
                        is_ile_val_l.append(rn1 in ("ILE", "VAL"))
                        next_is_pro_l.append(rn2 == "PRO")

                        triples_meta.append((
                            cid,
                            ch.id,
                            ch_strip,
                            residue_id(r1.resseq, r1.icode),
                            r1.icode,
                            r1.resseq_as_int(),
                            rn1,
                            altloc.strip(),
                        ))
                if c0_l:
                    ca0_chunks.append(np.asarray(ca0_l, dtype=np.float64))
                    c0_chunks.append(np.asarray(c0_l, dtype=np.float64))
                    n1_chunks.append(np.asarray(n1_l, dtype=np.float64))
                    ca1_chunks.append(np.asarray(ca1_l, dtype=np.float64))
                    c1_chunks.append(np.asarray(c1_l, dtype=np.float64))
                    n2_chunks.append(np.asarray(n2_l, dtype=np.float64))
                    om_ca0_chunks.append(np.asarray(om_ca0_l, dtype=np.float64))
                    om_c0_chunks.append(np.asarray(om_c0_l, dtype=np.float64))
                    om_n1_chunks.append(np.asarray(om_n1_l, dtype=np.float64))
                    om_ca1_chunks.append(np.asarray(om_ca1_l, dtype=np.float64))
                    is_gly_chunks.append(np.asarray(is_gly_l, dtype=np.bool_))
                    is_pro_chunks.append(np.asarray(is_pro_l, dtype=np.bool_))
                    is_ile_val_chunks.append(np.asarray(is_ile_val_l, dtype=np.bool_))
                    next_is_pro_chunks.append(np.asarray(next_is_pro_l, dtype=np.bool_))
        cid_all_pure_sorted[cid] = can_skip_sort

    if not triples_meta:
        return []

    n_triples = len(triples_meta)
    bucket_n = _next_bucket(n_triples)
    cat_pad = lambda chunks, dt: jnp.asarray(_pad_to(np.concatenate(chunks).astype(dt, copy=False), bucket_n))

    phi_arr, psi_arr, score_arr, _, eval_arr = rama_eval_batch_jax(
        cat_pad(ca0_chunks, np.float64),
        cat_pad(c0_chunks, np.float64),
        cat_pad(n1_chunks, np.float64),
        cat_pad(ca1_chunks, np.float64),
        cat_pad(c1_chunks, np.float64),
        cat_pad(n2_chunks, np.float64),
        cat_pad(om_ca0_chunks, np.float64),
        cat_pad(om_c0_chunks, np.float64),
        cat_pad(om_n1_chunks, np.float64),
        cat_pad(om_ca1_chunks, np.float64),
        cat_pad(is_gly_chunks, np.bool_),
        cat_pad(is_pro_chunks, np.bool_),
        cat_pad(is_ile_val_chunks, np.bool_),
        cat_pad(next_is_pro_chunks, np.bool_),
        tables["rama_tables"],
    )

    phi_list = np.asarray(phi_arr[:n_triples], dtype=np.float64).tolist()
    psi_list = np.asarray(psi_arr[:n_triples], dtype=np.float64).tolist()
    score_list = np.asarray(score_arr[:n_triples], dtype=np.float64).tolist()
    eval_list = np.asarray(eval_arr[:n_triples], dtype=np.int32).tolist()
    eval_names = ("OUTLIER", "Allowed", "Favored")

    if all(cid_all_pure_sorted.values()):
        return [
            (ch_strip, resid_str, rn1, alt_strip, eval_names[ev], phi, psi, sc)
            for (_, _, ch_strip, resid_str, _, _, rn1, alt_strip), ev, phi, psi, sc in zip(
                triples_meta, eval_list, phi_list, psi_list, score_list
            )
        ]

    by_cid = {cid: [] for cid in chain_ids_ordered}
    for i, ((cid, ch_id, ch_strip, resid_str, icode, resseq_int, rn1, alt_strip), ev, phi, psi, sc) in enumerate(
        zip(triples_meta, eval_list, phi_list, psi_list, score_list)
    ):
        sort_key = (ch_id, resseq_int, icode, alt_strip)
        rec = (ch_strip, resid_str, rn1, alt_strip, eval_names[ev], phi, psi, sc)
        by_cid[cid].append((sort_key, i, rec))

    out = []
    for cid in chain_ids_ordered:
        items = by_cid[cid]
        if not cid_all_pure_sorted.get(cid, False):
            items.sort(key=lambda x: (x[0], x[1]))
        out.extend(x[2] for x in items)
    return out


def _construct_complete_sidechain(rg):
    """
    Exact replica of rotalyze.construct_complete_sidechain(residue_group)
    (rotalyze.py:538-558) on a non-H hierarchy.
    """
    atom_dict = {}
    complete_dict = {}
    for ag in rg.atom_groups():
        if len(ag.atoms()) == 0:
            continue
        for a in ag.atoms():
            atom_dict[a.name] = a
        complete_dict[ag.altloc] = dict(atom_dict)
    if len(complete_dict) > 0:
        return complete_dict
    return atom_dict


def rotamers(hierarchy):
    """
    Every rotalyze result as (chain, resid, resname, altloc, evaluation, name, score).
    Accepts a JaxHierarchy, an iotbx.pdb.hierarchy.root, or a path to a PDB/mmCIF file.
    Evaluates all sidechains across the entire model in a single vectorized JAX/GPU batch.
    """
    h = _ensure_hierarchy(hierarchy)
    flat = _ensure_flat_index(h)
    xyz_all = flat["xyz"] if flat is not None else None
    chains_flat = flat["chains"] if flat is not None else {}
    smeta = _get_static_meta()
    chi_slots = smeta["chi_slots"]
    chi_req = smeta["chi_req"]
    unit_pts_np = smeta["unit_pts"]

    tables = get_tables()
    aa_to_idx = tables["aa_to_idx"]
    chi_atoms_map = tables["chi_atoms"]
    box_names = tables["box_names"]
    eval_names = ("OUTLIER", "Allowed", "Favored")

    model = h.models()[0]
    meta = []
    p0_chunks, p1_chunks, p2_chunks, p3_chunks = [], [], [], []
    chi_valid_chunks = []
    aa_idx_chunks = []
    is_sym_chunks = []

    unit_pts = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    for ch in model.chains():
        ce = chains_flat.get(id(ch))
        if ce is not None:
            r_def = ce["rota_def_idx"]
            cand = np.flatnonzero(r_def >= 0)
            if len(cand) == 0:
                continue
            def_c = r_def[cand]
            slots_c = chi_slots[def_c]
            req_c = chi_req[def_c]
            s_idx = ce["slot_idx"]
            a_idx = s_idx[cand[:, None, None], slots_c]
            complete = np.all((a_idx >= 0) | (~req_c[:, :, None]), axis=(1, 2))
            if not np.any(complete):
                continue
            cand = cand[complete]
            a_idx = a_idx[complete]
            req_c = req_c[complete]
            aa_c = ce["rota_aa_idx"][cand]
            sym_c = (ce["flags"][cand] & 64) != 0

            pts = xyz_all[np.maximum(a_idx, 0)]
            pts = np.where(req_c[:, :, None, None], pts, unit_pts_np[None, None, :, :])

            p0_chunks.append(pts[:, :, 0, :])
            p1_chunks.append(pts[:, :, 1, :])
            p2_chunks.append(pts[:, :, 2, :])
            p3_chunks.append(pts[:, :, 3, :])
            chi_valid_chunks.append(req_c)
            aa_idx_chunks.append(aa_c)
            is_sym_chunks.append(sym_c)

            cid_strip = ce["cid_strip"]
            resids = ce["resids"]
            resnames = ce["resnames"]
            cand_list = cand.tolist()
            aa_list_c = aa_c.tolist()
            meta.extend((cid_strip, resids[k], resnames[k], "", ai) for k, ai in zip(cand_list, aa_list_c))
        else:
            p0_l, p1_l, p2_l, p3_l = [], [], [], []
            chi_valid_l, aa_idx_l, is_sym_l = [], [], []
            cid_strip = ch.id.strip()
            for rg in ch.residue_groups():
                resid_str = residue_id(rg.resseq, rg.icode)
                all_dict = _construct_complete_sidechain(rg)
                for ag in rg.atom_groups():
                    resname_strip = ag.resname.strip()
                    resname_lower = resname_strip.lower()
                    if resname_strip not in COMMON_AMINO_ACIDS:
                        continue
                    if resname_lower in ("ala", "gly"):
                        continue
                    eval_res = "met" if resname_lower == "mse" else resname_lower
                    aa_i = aa_to_idx.get(eval_res)
                    if aa_i is None:
                        continue
                    atom_dict = all_dict.get(ag.altloc, {})
                    by_name = {k.strip(): v for k, v in atom_dict.items()}
                    chi_defs = chi_atoms_map[resname_lower]

                    c_p0, c_p1, c_p2, c_p3 = [], [], [], []
                    c_val = []
                    incomplete = False
                    for c_atoms in chi_defs:
                        a0 = by_name.get(c_atoms[0])
                        a1 = by_name.get(c_atoms[1])
                        a2 = by_name.get(c_atoms[2])
                        a3 = by_name.get(c_atoms[3])
                        if a0 is None or a1 is None or a2 is None or a3 is None:
                            incomplete = True
                            break
                        c_p0.append(a0.xyz)
                        c_p1.append(a1.xyz)
                        c_p2.append(a2.xyz)
                        c_p3.append(a3.xyz)
                        c_val.append(True)
                    if incomplete:
                        continue

                    while len(c_val) < 4:
                        c_p0.append(unit_pts[0])
                        c_p1.append(unit_pts[1])
                        c_p2.append(unit_pts[2])
                        c_p3.append(unit_pts[3])
                        c_val.append(False)

                    p0_l.append(c_p0)
                    p1_l.append(c_p1)
                    p2_l.append(c_p2)
                    p3_l.append(c_p3)
                    chi_valid_l.append(c_val)
                    aa_idx_l.append(aa_i)
                    is_sym_l.append(resname_lower in SYM_AA)
                    meta.append((cid_strip, resid_str, resname_strip, ag.altloc.strip(), aa_i))
            if p0_l:
                p0_chunks.append(np.asarray(p0_l, dtype=np.float64))
                p1_chunks.append(np.asarray(p1_l, dtype=np.float64))
                p2_chunks.append(np.asarray(p2_l, dtype=np.float64))
                p3_chunks.append(np.asarray(p3_l, dtype=np.float64))
                chi_valid_chunks.append(np.asarray(chi_valid_l, dtype=np.bool_))
                aa_idx_chunks.append(np.asarray(aa_idx_l, dtype=np.int32))
                is_sym_chunks.append(np.asarray(is_sym_l, dtype=np.bool_))

    if not meta:
        return []

    n_rota = len(meta)
    bucket_n = _next_bucket(n_rota)
    cat_pad = lambda chunks, dt: jnp.asarray(_pad_to(np.concatenate(chunks).astype(dt, copy=False), bucket_n))

    score_arr, eval_arr, box_arr = rotamer_eval_batch_jax(
        cat_pad(p0_chunks, np.float64),
        cat_pad(p1_chunks, np.float64),
        cat_pad(p2_chunks, np.float64),
        cat_pad(p3_chunks, np.float64),
        cat_pad(chi_valid_chunks, np.bool_),
        cat_pad(aa_idx_chunks, np.int32),
        cat_pad(is_sym_chunks, np.bool_),
        tables["rota_flat"],
        tables["rota_offset"],
        tables["rota_ndim"],
        tables["rota_min"],
        tables["rota_max"],
        tables["rota_nbins"],
        tables["rota_dowrap"],
        tables["rota_wbin"],
        tables["box_low"],
        tables["box_high"],
        tables["box_valid"],
    )

    score_list = np.asarray(score_arr[:n_rota], dtype=np.float64).tolist()
    eval_list = np.asarray(eval_arr[:n_rota], dtype=np.int32).tolist()
    box_list = np.asarray(box_arr[:n_rota], dtype=np.int32).tolist()

    return [
        (
            cid_strip,
            resid_str,
            resname_strip,
            alt_strip,
            eval_names[ev_code],
            "OUTLIER" if ev_code == 0 else (box_names[aa_i][b_idx] if b_idx >= 0 else "UNCLASSIFIED"),
            sc,
        )
        for (cid_strip, resid_str, resname_strip, alt_strip, aa_i), ev_code, b_idx, sc in zip(
            meta, eval_list, box_list, score_list
        )
    ]


def _rotate_point_around_axis_jax(axis_p1, axis_p2, point, angle_deg):
    """
    Exact vectorized replica of scitbx::math::rotate_point_around_axis
    (scitbx/math/rotate_around_axis.h:58-97).
    axis_p1, axis_p2, point: (N, 3) float64 arrays.
    angle_deg: (N,) float64 array in degrees.
    """
    angle_rad = angle_deg * (jnp.pi / 180.0)
    l = axis_p2 - axis_p1
    dlsq = jnp.sum(l * l, axis=-1, keepdims=True)
    safe_dlsq = jnp.where(dlsq == 0.0, 1.0, dlsq)
    dl = jnp.sqrt(safe_dlsq)
    ca = jnp.cos(angle_rad)[:, None]
    dsa = jnp.sin(angle_rad)[:, None] / dl
    oca = (1.0 - ca) / safe_dlsq
    m = point - axis_p1
    l_dot_m = jnp.sum(l * m, axis=-1, keepdims=True)
    l_cross_m = jnp.cross(l, m)
    return axis_p1 + m * ca + l * (l_dot_m * oca) + l_cross_m * dsa


def _construct_fourth_jax(res0, res1, res2, dist, angle, dihedral):
    """
    Vectorized replica of mmtbx.validation.cbetadev.construct_fourth
    (cbetadev.py:322-358).
    """
    a = res2 - res1
    b = res0 - res1
    c = jnp.cross(a, b)
    cmag = jnp.sqrt(jnp.sum(c * c, axis=-1, keepdims=True))
    c = jnp.where(cmag > 1e-6, c * (dist[:, None] / jnp.where(cmag > 1e-6, cmag, 1.0)), c)
    d = c + res2
    new_d = _rotate_point_around_axis_jax(res1, res2, d, dihedral - 90.0)
    a2 = new_d - res2
    b2 = res1 - res2
    c2 = jnp.cross(a2, b2)
    cmag2 = jnp.sqrt(jnp.sum(c2 * c2, axis=-1, keepdims=True))
    c2 = jnp.where(cmag2 > 1e-6, c2 * (dist[:, None] / jnp.where(cmag2 > 1e-6, cmag2, 1.0)), c2)
    b_axis = c2 + res2
    return _rotate_point_around_axis_jax(res2, b_axis, new_d, 90.0 - angle)


@jax.jit
def _cbeta_batch_kernel(res_n, res_ca, res_c, res_cb, res_type, is_common_aa, is_d_aa):
    """
    Vectorized GPU kernel for C-beta deviation and dihedral_NABB
    (mmtbx/validation/cbetadev.py:216-358).
    res_type: 0=default, 1=ALA/DAL, 2=PRO/DPR, 3=VAL/THR/ILE/DVA/DTH/DIL.
    """
    # Chiral volume at CA: (N - CA) . ((CB - CA) x (C - CA))
    d01 = res_n - res_ca
    d02 = res_cb - res_ca
    d03 = res_c - res_ca
    chiral_vol = jnp.sum(d01 * jnp.cross(d02, d03), axis=-1)

    dist_lut = jnp.array([1.530, 1.536, 1.530, 1.540], dtype=jnp.float64)
    ang_cab_lut = jnp.array([110.1, 110.1, 112.2, 109.1], dtype=jnp.float64)
    dih_ncab_lut = jnp.array([122.8, 122.9, 115.1, 123.4], dtype=jnp.float64)
    ang_nab_lut = jnp.array([110.5, 110.6, 103.0, 111.5], dtype=jnp.float64)
    dih_cnab_lut = jnp.array([-122.6, -122.6, -120.7, -122.0], dtype=jnp.float64)

    dist = dist_lut[res_type]
    ang_cab = ang_cab_lut[res_type]
    dih_ncab = dih_ncab_lut[res_type]
    ang_nab = ang_nab_lut[res_type]
    dih_cnab = dih_cnab_lut[res_type]

    flip_d = (~is_common_aa) & (is_d_aa | (chiral_vol > 0.0))
    dih_ncab = jnp.where(flip_d, -dih_ncab, dih_ncab)
    dih_cnab = jnp.where(flip_d, -dih_cnab, dih_cnab)

    beta_ncab = _construct_fourth_jax(res_n, res_c, res_ca, dist, ang_cab, dih_ncab)
    beta_cnab = _construct_fourth_jax(res_c, res_n, res_ca, dist, ang_nab, dih_cnab)
    betaxyz = 0.5 * (beta_ncab + beta_cnab)

    diff_ca = betaxyz - res_ca
    betadist = jnp.sqrt(jnp.sum(diff_ca * diff_ca, axis=-1))
    safe_bdist = jnp.where(betadist > 0.0, betadist, 1.0)
    betaxyz = jnp.where(
        (betadist > 0.0)[:, None],
        res_ca + diff_ca * (dist / safe_bdist)[:, None],
        betaxyz,
    )

    dev_vec = res_cb - betaxyz
    deviation = jnp.sqrt(jnp.sum(dev_vec * dev_vec, axis=-1))
    dihedral_nabb = dihedral_angles_jax(res_n, res_ca, betaxyz, res_cb)
    return deviation, dihedral_nabb


def cbeta(hierarchy):
    """
    Every cbetadev result as (chain, resid, resname, altloc, deviation, dihedral NABB, occupancy).
    Vectorized JAX/GPU implementation bit-exact with mmtbx.validation.cbetadev.
    """
    h = _ensure_hierarchy(hierarchy)
    flat = _ensure_flat_index(h)
    xyz_all = flat["xyz"] if flat is not None else None
    occ_all = flat["occ"] if flat is not None else None
    chains_flat = flat["chains"] if flat is not None else {}

    cab_tables = get_cablam_tables()
    common_aa = cab_tables["common_aa"]
    d_aa = cab_tables["d_aa"]

    model = h.models()[0]
    meta = []
    n_chunks, ca_chunks, c_chunks, cb_chunks = [], [], [], []
    rtype_chunks, common_chunks, daa_chunks = [], [], []

    for ch in model.chains():
        ce = chains_flat.get(id(ch))
        if ce is not None:
            s_idx = ce["slot_idx"]
            n_i = s_idx[:, 0]
            ca_i = s_idx[:, 1]
            c_i = s_idx[:, 2]
            cb_i = s_idx[:, 4]
            rt = ce["cbeta_rtype"]
            cand = np.flatnonzero((rt >= 0) & (n_i >= 0) & (ca_i >= 0) & (c_i >= 0) & (cb_i >= 0))
            if len(cand) == 0:
                continue
            n_chunks.append(xyz_all[n_i[cand]])
            ca_chunks.append(xyz_all[ca_i[cand]])
            c_chunks.append(xyz_all[c_i[cand]])
            cb_chunks.append(xyz_all[cb_i[cand]])
            rtype_chunks.append(rt[cand])
            fl_c = ce["flags"][cand]
            common_chunks.append((fl_c & 1) != 0)
            daa_chunks.append((fl_c & 4) != 0)

            cid_strip = ce["cid_strip"]
            resids = ce["resids"]
            resnames = ce["resnames"]
            cand_list = cand.tolist()
            occ_list = occ_all[cb_i[cand]].tolist()
            meta.extend((cid_strip, resids[k], resnames[k], "", oc) for k, oc in zip(cand_list, occ_list))
        else:
            n_l, ca_l, c_l, cb_l = [], [], [], []
            rtype_l, common_l, daa_l = [], [], []
            cid_strip = ch.id.strip()
            for rg in ch.residue_groups():
                resid_str = residue_id(rg.resseq, rg.icode)
                for i_cf, cf in enumerate(rg.conformers()):
                    for res in cf.residues():
                        rn_strip = res.resname.strip()
                        if rn_strip == "GLY":
                            continue
                        is_alt_conf = False
                        rel = {}
                        for a in res.atoms():
                            if a.name in (" CA ", " N  ", " C  ", " CB "):
                                rel[a.name] = a
                                ag = a.parent()
                                if ag is not None and len(ag.altloc) != 0:
                                    is_alt_conf = True
                        if (i_cf == 0 or is_alt_conf) and len(rel) == 4:
                            res_ca = rel[" CA "]
                            res_n = rel[" N  "]
                            res_c = rel[" C  "]
                            res_cb = rel[" CB "]
                            altchar = cf.altloc.strip() if is_alt_conf else ""
                            if rn_strip in ("ALA", "DAL"):
                                rt_v = 1
                            elif rn_strip in ("PRO", "DPR"):
                                rt_v = 2
                            elif rn_strip in ("VAL", "THR", "ILE", "DVA", "DTH", "DIL"):
                                rt_v = 3
                            else:
                                rt_v = 0
                            n_l.append(res_n.xyz)
                            ca_l.append(res_ca.xyz)
                            c_l.append(res_c.xyz)
                            cb_l.append(res_cb.xyz)
                            rtype_l.append(rt_v)
                            common_l.append(rn_strip in common_aa)
                            daa_l.append(rn_strip in d_aa)
                            meta.append((cid_strip, resid_str, rn_strip, altchar, float(res_cb.occ)))
            if n_l:
                n_chunks.append(np.asarray(n_l, dtype=np.float64))
                ca_chunks.append(np.asarray(ca_l, dtype=np.float64))
                c_chunks.append(np.asarray(c_l, dtype=np.float64))
                cb_chunks.append(np.asarray(cb_l, dtype=np.float64))
                rtype_chunks.append(np.asarray(rtype_l, dtype=np.int32))
                common_chunks.append(np.asarray(common_l, dtype=np.bool_))
                daa_chunks.append(np.asarray(daa_l, dtype=np.bool_))

    if not meta:
        return []

    n_items = len(meta)
    bucket_n = _next_bucket(n_items)
    cat_pad = lambda chunks, dt: jnp.asarray(_pad_to(np.concatenate(chunks).astype(dt, copy=False), bucket_n))

    dev_arr, dih_arr = _cbeta_batch_kernel(
        cat_pad(n_chunks, np.float64),
        cat_pad(ca_chunks, np.float64),
        cat_pad(c_chunks, np.float64),
        cat_pad(cb_chunks, np.float64),
        cat_pad(rtype_chunks, np.int32),
        cat_pad(common_chunks, np.bool_),
        cat_pad(daa_chunks, np.bool_),
    )
    dev_list = np.asarray(dev_arr[:n_items], dtype=np.float64).tolist()
    dih_list = np.asarray(dih_arr[:n_items], dtype=np.float64).tolist()

    return [
        (cid_strip, resid_str, rn_strip, altchar, dev, dih, occ)
        for (cid_strip, resid_str, rn_strip, altchar, occ), dev, dih in zip(meta, dev_list, dih_list)
    ]


def _local_altloc_from_atoms(atoms):
    altloc = ""
    for a in atoms:
        ag = a.parent()
        if ag is not None and ag.altloc != "":
            altloc = ag.altloc
    return altloc.strip()


def omegas(hierarchy):
    """
    Every peptide omegalyze measures, in every conformer, before its filter:
    (chain, resid, resname, altloc, previous resid, previous altloc, omega,
    type, conformer altloc, the peptide's altloc).
    Vectorized JAX/GPU implementation bit-exact with remolprobity.residues.omegas.
    """
    h = _ensure_hierarchy(hierarchy)
    flat = _ensure_flat_index(h)
    xyz_all = flat["xyz"] if flat is not None else None
    chains_flat = flat["chains"] if flat is not None else {}

    all_aa = get_cablam_tables()["all_aa"]
    model = h.models()[0]

    meta = []
    ca0_chunks, c0_chunks, n1_chunks, ca1_chunks = [], [], [], []

    for ch in model.chains():
        ce = chains_flat.get(id(ch))
        if ce is not None:
            bb_aa = np.flatnonzero(((ce["flags"] & 2) != 0) & ce["has_bb"])
            if len(bb_aa) < 2:
                continue
            r0 = bb_aa[:-1]
            r1 = bb_aa[1:]
            s_idx = ce["slot_idx"]
            ca0_i = s_idx[r0, 1]
            c0_i = s_idx[r0, 2]
            n1_i = s_idx[r1, 0]
            ca1_i = s_idx[r1, 1]
            valid = (ca0_i >= 0) & (c0_i >= 0) & (n1_i >= 0) & (ca1_i >= 0)
            if not np.any(valid):
                continue
            r0, r1 = r0[valid], r1[valid]
            ca0_i, c0_i, n1_i, ca1_i = ca0_i[valid], c0_i[valid], n1_i[valid], ca1_i[valid]
            valid2 = np.sum((xyz_all[c0_i] - xyz_all[n1_i]) ** 2, axis=-1) < 4.0
            if not np.any(valid2):
                continue
            r0, r1 = r0[valid2], r1[valid2]
            ca0_i, c0_i, n1_i, ca1_i = ca0_i[valid2], c0_i[valid2], n1_i[valid2], ca1_i[valid2]

            ca0_chunks.append(xyz_all[ca0_i])
            c0_chunks.append(xyz_all[c0_i])
            n1_chunks.append(xyz_all[n1_i])
            ca1_chunks.append(xyz_all[ca1_i])

            cid_strip = ce["cid_strip"]
            resids = ce["resids"]
            resnames = ce["resnames"]
            r0_list = r0.tolist()
            r1_list = r1.tolist()
            meta.extend(
                (cid_strip, resids[k1], resnames[k1], "", resids[k0], "", "", "")
                for k0, k1 in zip(r0_list, r1_list)
            )
        else:
            ca0_l, c0_l, n1_l, ca1_l = [], [], [], []
            cid_strip = ch.id.strip()
            bb_ch = _select_backbone_chain(ch)
            for conf in bb_ch.conformers():
                conf_alt = conf.altloc
                two = []
                for res in conf.residues():
                    rn = res.resname.strip()
                    if rn not in all_aa:
                        continue
                    if two:
                        prev_c = two[-1].find_atom_by(" C  ")
                        curr_n = res.find_atom_by(" N  ")
                        if prev_c is None or curr_n is None:
                            two = [res]
                            continue
                        dx = prev_c.xyz[0] - curr_n.xyz[0]
                        dy = prev_c.xyz[1] - curr_n.xyz[1]
                        dz = prev_c.xyz[2] - curr_n.xyz[2]
                        if dx * dx + dy * dy + dz * dz >= 4.0:
                            two = [res]
                            continue
                    two.append(res)
                    if len(two) > 2:
                        two.pop(0)
                    if len(two) < 2:
                        continue

                    r0, r1 = two[0], two[1]
                    prev_ca = prev_c = this_n = this_ca = None
                    for a in r0.atoms():
                        if a.name == " CA ":
                            prev_ca = a
                        elif a.name == " C  ":
                            prev_c = a
                    for a in r1.atoms():
                        if a.name == " N  ":
                            this_n = a
                        elif a.name == " CA ":
                            this_ca = a
                    if prev_ca is None or prev_c is None or this_n is None or this_ca is None:
                        continue

                    atoms = [prev_ca, prev_c, this_n, this_ca]
                    ca0_l.append(prev_ca.xyz)
                    c0_l.append(prev_c.xyz)
                    n1_l.append(this_n.xyz)
                    ca1_l.append(this_ca.xyz)
                    meta.append((
                        cid_strip,
                        residue_id(r1.resseq, r1.icode),
                        r1.resname.strip(),
                        _local_altloc_from_atoms(atoms[2:]),
                        residue_id(r0.resseq, r0.icode),
                        _local_altloc_from_atoms(atoms[0:2]),
                        conf_alt,
                        _local_altloc_from_atoms(atoms),
                    ))
            if ca0_l:
                ca0_chunks.append(np.asarray(ca0_l, dtype=np.float64))
                c0_chunks.append(np.asarray(c0_l, dtype=np.float64))
                n1_chunks.append(np.asarray(n1_l, dtype=np.float64))
                ca1_chunks.append(np.asarray(ca1_l, dtype=np.float64))

    if not meta:
        return []

    n_items = len(meta)
    bucket_n = _next_bucket(n_items)
    cat_pad = lambda chunks, dt: jnp.asarray(_pad_to(np.concatenate(chunks).astype(dt, copy=False), bucket_n))

    omega_arr = dihedral_angles_jax(
        cat_pad(ca0_chunks, np.float64),
        cat_pad(c0_chunks, np.float64),
        cat_pad(n1_chunks, np.float64),
        cat_pad(ca1_chunks, np.float64),
    )
    omega_list = np.asarray(omega_arr[:n_items], dtype=np.float64).tolist()

    return [
        (
            cid_strip,
            res1_id,
            rn1,
            alt1,
            res0_id,
            alt0,
            om,
            "Cis" if -30.0 < om < 30.0 else ("Trans" if (om < -150.0 or om > 150.0) else "Twisted"),
            conf_alt,
            pep_alt,
        )
        for (cid_strip, res1_id, rn1, alt1, res0_id, alt0, conf_alt, pep_alt), om in zip(meta, omega_list)
    ]


def _ndt_interp_3d_jax(tables_3d, t_idx, x0, x1, x2, min2, nbins2):
    """
    Exact vectorized replica of NDimTable.valueAt for 3D CaBLAM tables
    (n_dim_table.py:165-196).
    tables_3d: (4, 120, 120, N2) float64 array where N2 is 120 (pep) or 60 (CA).
    """
    coords = jnp.stack([x0, x1, x2], axis=-1)
    min_v = jnp.array([-180.0, -180.0, min2], dtype=jnp.float64)
    w_b = jnp.array([3.0, 3.0, 3.0], dtype=jnp.float64)
    n_b = jnp.array([120, 120, nbins2], dtype=jnp.int32)
    n_b_f = n_b.astype(jnp.float64)

    va_home = jnp.minimum(jnp.floor((coords - min_v) / w_b), n_b_f - 1.0).astype(jnp.int32)
    va_home_ctr = min_v + w_b * (va_home.astype(jnp.float64) + 0.5)
    va_neighbor = jnp.where(coords < va_home_ctr, va_home - 1, va_home + 1)
    va_contrib = jnp.abs((coords - va_home_ctr) / w_b)

    # All 3 dimensions have doWrap=True in CaBLAM tables
    def _wrap_lim(b_val, nb_d):
        w = jnp.mod(b_val, nb_d)
        w = jnp.where(w < 0, w + nb_d, w)
        return jnp.clip(w, 0, nb_d - 1).astype(jnp.int32)

    h0 = _wrap_lim(va_home[:, 0], 120)
    h1 = _wrap_lim(va_home[:, 1], 120)
    h2 = _wrap_lim(va_home[:, 2], nbins2)
    n0 = _wrap_lim(va_neighbor[:, 0], 120)
    n1 = _wrap_lim(va_neighbor[:, 1], 120)
    n2 = _wrap_lim(va_neighbor[:, 2], nbins2)

    c0 = va_contrib[:, 0]
    c1 = va_contrib[:, 1]
    c2 = va_contrib[:, 2]

    res = jnp.zeros(x0.shape[0], dtype=jnp.float64)
    for b_idx in range(8):
        i0 = h0 if (b_idx & 1) == 0 else n0
        i1 = h1 if (b_idx & 2) == 0 else n1
        i2 = h2 if (b_idx & 4) == 0 else n2
        w0 = (1.0 - c0) if (b_idx & 1) == 0 else c0
        w1 = (1.0 - c1) if (b_idx & 2) == 0 else c1
        w2 = (1.0 - c2) if (b_idx & 4) == 0 else c2
        res = res + (w0 * w1 * w2) * tables_3d[t_idx, i0, i1, i2]
    return res


@jax.jit
def _cablam_geom_batch_kernel(
    ca0, ca1, ca2, ca3, ca4, o1, o2, c1, n2,
    has_ca_3, has_mu_in, has_mu_out, has_nu_in, has_omega
):
    """
    Vectorized GPU kernel for CaBLAM virtual angles/dihedrals (mu_in, mu_out,
    nu, ca_virtual, omega) matching mmtbx.validation.cablam.cablam_math.
    """
    mu_in = dihedral_angles_jax(ca0, ca1, ca2, ca3)
    mu_out = dihedral_angles_jax(ca1, ca2, ca3, ca4)

    # ca_virtual = scitbx::math::angle(ca1, ca2, ca3)
    v1 = ca1 - ca2
    v3 = ca3 - ca2
    n1_sq = jnp.sum(v1 * v1, axis=-1)
    n3_sq = jnp.sum(v3 * v3, axis=-1)
    denom = jnp.sqrt(n1_sq * n3_sq)
    safe_denom = jnp.where(denom > 0.0, denom, 1.0)
    cos_ang = jnp.clip(jnp.sum(v1 * v3, axis=-1) / safe_denom, -1.0, 1.0)
    ca_virt = jnp.arccos(cos_ang) * (180.0 / jnp.pi)

    # perptersect(a1, a2, b1): projection of b1 onto line a1..a2
    def _perptersect(a1, a2, b1):
        u = a2 - a1
        u_sq = jnp.sum(u * u, axis=-1, keepdims=True)
        valid = (u_sq[:, 0] > 0.0)
        safe_u_sq = jnp.where(u_sq > 0.0, u_sq, 1.0)
        t = jnp.sum((b1 - a1) * u, axis=-1, keepdims=True) / safe_u_sq
        return a1 + t * u, valid

    x1, v_x1 = _perptersect(ca1, ca2, o1)
    x2, v_x2 = _perptersect(ca2, ca3, o2)
    valid_nu = has_nu_in & v_x1 & v_x2
    nu = dihedral_angles_jax(o1, x1, x2, o2)
    omega = dihedral_angles_jax(ca1, c1, n2, ca2)

    return mu_in, mu_out, nu, ca_virt, omega, valid_nu


@jax.jit
def _cablam_eval_batch_kernel(mu_in, mu_out, nu, ca_virt, t_idx, pep_tables, ca_tables):
    """Vectorized 3D trilinear contour evaluation for CaBLAM and CA-geom."""
    cab_score = _ndt_interp_3d_jax(pep_tables, t_idx, mu_in, mu_out, nu, -180.0, 120)
    ca_score = _ndt_interp_3d_jax(ca_tables, t_idx, mu_in, mu_out, ca_virt, 0.0, 60)
    return cab_score, ca_score


@jax.jit
def _cablam_fused_batch_kernel(
    ca0, ca1, ca2, ca3, ca4, o1, o2, c1, n2,
    has_nu_in, has_omega, base_tidx, pep_tables, ca_tables
):
    """Fused virtual geometry + 3D contour evaluation for pure main-conformation chains."""
    mu_in = dihedral_angles_jax(ca0, ca1, ca2, ca3)
    mu_out = dihedral_angles_jax(ca1, ca2, ca3, ca4)

    v1 = ca1 - ca2
    v3 = ca3 - ca2
    n1_sq = jnp.sum(v1 * v1, axis=-1)
    n3_sq = jnp.sum(v3 * v3, axis=-1)
    denom = jnp.sqrt(n1_sq * n3_sq)
    safe_denom = jnp.where(denom > 0.0, denom, 1.0)
    cos_ang = jnp.clip(jnp.sum(v1 * v3, axis=-1) / safe_denom, -1.0, 1.0)
    ca_virt = jnp.arccos(cos_ang) * (180.0 / jnp.pi)

    def _perptersect(a1, a2, b1):
        u = a2 - a1
        u_sq = jnp.sum(u * u, axis=-1, keepdims=True)
        valid = u_sq[:, 0] > 0.0
        safe_u_sq = jnp.where(u_sq > 0.0, u_sq, 1.0)
        t = jnp.sum((b1 - a1) * u, axis=-1, keepdims=True) / safe_u_sq
        return a1 + t * u, valid

    x1, v_x1 = _perptersect(ca1, ca2, o1)
    x2, v_x2 = _perptersect(ca2, ca3, o2)
    valid_nu = has_nu_in & v_x1 & v_x2
    nu = jnp.where(valid_nu, dihedral_angles_jax(o1, x1, x2, o2), 0.0)
    omega = dihedral_angles_jax(ca1, c1, n2, ca2)

    is_cis_pro = (base_tidx == 2) & has_omega & (omega >= -90.0) & (omega <= 90.0)
    t_idx = jnp.where(is_cis_pro, 3, base_tidx)

    cab_score = _ndt_interp_3d_jax(pep_tables, t_idx, mu_in, mu_out, nu, -180.0, 120)
    ca_score = _ndt_interp_3d_jax(ca_tables, t_idx, mu_in, mu_out, ca_virt, 0.0, 60)
    return cab_score, ca_score, valid_nu


def cablam(hierarchy, owned):
    """
    CaBLAM in each chain object of hierarchy on its own, as cablamalyze
    makes it: (records, classes).
    Vectorized JAX/GPU implementation bit-exact with remolprobity.residues.cablam.
    """
    h = _ensure_hierarchy(hierarchy)
    flat = _ensure_flat_index(h)
    xyz_all = flat["xyz"] if flat is not None else None
    chains_flat = flat["chains"] if flat is not None else {}

    cab_tables = get_cablam_tables()
    common_aa = cab_tables["common_aa"]
    all_aa = cab_tables["all_aa"]
    all_na = cab_tables["all_na"]
    d_to_l = cab_tables["d_to_l"]

    records, classes = [], []
    model = h.models()[0]
    model_id = model.id

    # Fast path for pure main-conformation chains + fallback for chains with altlocs
    pure_owned_lists = []
    pure_eval_targets = []
    f_ca0, f_ca1, f_ca2, f_ca3, f_ca4 = [], [], [], [], []
    f_o1, f_o2, f_c1, f_n2 = [], [], [], []
    f_has_nu, f_has_om, f_tidx = [], [], []

    chains_work = []
    geom_nodes = []
    ca0_l, ca1_l, ca2_l, ca3_l, ca4_l = [], [], [], [], []
    o1_l, o2_l, c1_l, n2_l = [], [], [], []
    has_ca3_l, has_mu_in_l, has_mu_out_l, has_nu_l, has_om_l = [], [], [], [], []

    z0 = (0.0, 0.0, 0.0)
    u1, u2, u3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    u1_np = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u3_np = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for chain in model.chains():
        ce = chains_flat.get(id(chain))
        if ce is not None:
            cid = ce["cid_strip"]
            resids = ce["resids"]
            resnames = ce["resnames"]
            cab_cls_all = ce["cab_classes"]
            mine = [owned(cid, rid) for rid in resids]
            if not any(mine):
                continue
            if all(mine):
                classes.extend(cab_cls_all)
                mine_arr = None
            else:
                classes.extend(c for c, m_flag in zip(cab_cls_all, mine) if m_flag)
                mine_arr = np.array(mine, dtype=bool)

            m = ce["m"]
            s_idx = ce["slot_idx"]
            n_i = s_idx[:, 0]
            ca_i = s_idx[:, 1]
            c_i = s_idx[:, 2]
            o_i = s_idx[:, 3]
            has_ca = ca_i >= 0
            has_mc = has_ca & (c_i >= 0) & (n_i >= 0) & (o_i >= 0)

            if m >= 2:
                both_ca = has_ca[:-1] & has_ca[1:]
                both_mc = has_mc[:-1] & has_mc[1:]
                d_cn = np.sum((xyz_all[np.maximum(c_i[:-1], 0)] - xyz_all[np.maximum(n_i[1:], 0)]) ** 2, axis=-1) < 4.0
                d_caca = np.sum((xyz_all[np.maximum(ca_i[:-1], 0)] - xyz_all[np.maximum(ca_i[1:], 0)]) ** 2, axis=-1) < 20.25
                linked = both_ca & np.where(both_mc, d_cn, d_caca)
            else:
                linked = np.zeros(0, dtype=bool)

            L = np.concatenate([[False, False], linked, [False, False]])
            link_p1 = L[1 : m + 1]
            link_n1 = L[2 : m + 2]
            link_p2 = L[0:m]
            link_n2 = L[3 : m + 3]
            has_ca3 = has_ca & link_p1 & link_n1
            has_mu_in = has_ca3 & link_p2
            has_mu_out = has_ca3 & link_n2
            mc_p1 = np.concatenate([[False], has_mc[:-1]])
            mc_n1 = np.concatenate([has_mc[1:], [False]])
            has_nu = has_mc & link_p1 & link_n1 & mc_p1 & mc_n1
            has_om = has_mc & link_p1 & mc_p1

            owned_ca = np.flatnonzero(has_ca if mine_arr is None else (has_ca & mine_arr))
            if len(owned_ca) == 0:
                continue
            if not ce["is_sorted"]:
                ri_arr = ce["resseq_ints"]
                ic_list = ce["icodes"]
                order = sorted(range(len(owned_ca)), key=lambda idx: (int(ri_arr[owned_ca[idx]]), ic_list[owned_ca[idx]]))
                owned_ca = owned_ca[np.array(order, dtype=np.int64)]

            oca_list = owned_ca.tolist()
            ch_recs = [[cid, resids[k], resnames[k], "", None, None] for k in oca_list]
            pure_owned_lists.append(("pure", ch_recs))

            cab_t = ce["cab_tidx"]
            ev_mask = (cab_t[owned_ca] >= 0) & has_mu_in[owned_ca] & has_mu_out[owned_ca]
            if np.any(ev_mask):
                ev_pos = np.flatnonzero(ev_mask)
                ev_r = owned_ca[ev_pos]
                for pos in ev_pos.tolist():
                    pure_eval_targets.append(ch_recs[pos])
                f_ca0.append(xyz_all[ca_i[ev_r - 2]])
                f_ca1.append(xyz_all[ca_i[ev_r - 1]])
                f_ca2.append(xyz_all[ca_i[ev_r]])
                f_ca3.append(xyz_all[ca_i[ev_r + 1]])
                f_ca4.append(xyz_all[ca_i[ev_r + 2]])
                h_nu_e = has_nu[ev_r]
                h_om_e = has_om[ev_r]
                f_has_nu.append(h_nu_e)
                f_has_om.append(h_om_e)
                f_tidx.append(cab_t[ev_r])
                if np.all(h_nu_e):
                    f_o1.append(xyz_all[o_i[ev_r - 1]])
                    f_o2.append(xyz_all[o_i[ev_r]])
                else:
                    f_o1.append(np.where(h_nu_e[:, None], xyz_all[np.maximum(o_i[ev_r - 1], 0)], u1_np))
                    f_o2.append(np.where(h_nu_e[:, None], xyz_all[np.maximum(o_i[ev_r], 0)], u3_np))
                if np.all(h_om_e):
                    f_c1.append(xyz_all[c_i[ev_r - 1]])
                    f_n2.append(xyz_all[n_i[ev_r]])
                else:
                    f_c1.append(np.where(h_om_e[:, None], xyz_all[np.maximum(c_i[ev_r - 1], 0)], u1_np))
                    f_n2.append(np.where(h_om_e[:, None], xyz_all[np.maximum(n_i[ev_r], 0)], u3_np))
            continue

        cid = chain.id.strip()
        keys = [(cid, residue_id(rg.resseq, rg.icode)) for rg in chain.residue_groups()]
        mine = [owned(*k) for k in keys]
        if not any(mine):
            continue
        for rg, k, own in zip(chain.residue_groups(), keys, mine):
            if not own:
                continue
            ag = rg.atom_groups()[0]
            rn = ag.resname.strip().upper()
            if not rn:
                rn = None
            elif rn in d_to_l:
                rn = d_to_l[rn]
            elif len(ag.atoms()) == 1 and rn in ("A", "C", "G", "U", "DA", "DC", "DG", "DT", "T"):
                an = ag.atoms()[0].name.strip()
                if an not in ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"):
                    rn = None
            is_aa = int(rn in all_aa) if rn is not None else 0
            is_na = int(rn in all_na) if rn is not None else 0
            classes.append(k + (int(rn != "HOH"), is_aa, is_na, int(rn == "UNK")))

        conformer_ids = []
        results_by_conformer = {}
        all_keys = []
        all_keys_set = set()

        for conf in chain.conformers():
            cname = conf.altloc.strip()
            if cname == "":
                cname = " "
            if cname not in conformer_ids:
                conformer_ids.append(cname)
            cur_conf_res = {}
            results_by_conformer[cname] = cur_conf_res
            prev_node = None

            for res in conf.residues():
                sid = f"{model_id}{chain.id.rjust(2)}{res.resseq}{res.icode}"
                if sid not in all_keys_set:
                    all_keys_set.add(sid)
                    all_keys.append(sid)

                ca = res.find_atom_by(" CA ")
                c = res.find_atom_by(" C  ")
                n = res.find_atom_by(" N  ")
                o = res.find_atom_by(" O  ")
                has_ca = ca is not None
                has_mc = has_ca and (c is not None) and (n is not None) and (o is not None)

                node = {
                    "chain_id": chain.id,
                    "resseq": res.resseq,
                    "icode": res.icode,
                    "resname": res.resname,
                    "altloc": conf.altloc.strip(),
                    "has_ca": has_ca,
                    "has_mc": has_mc,
                    "ca": ca.xyz if ca is not None else None,
                    "c": c.xyz if c is not None else None,
                    "n": n.xyz if n is not None else None,
                    "o": o.xyz if o is not None else None,
                    "prev": None,
                    "next": None,
                    "mu_in": None,
                    "mu_out": None,
                    "nu": None,
                    "ca_virt": None,
                    "omega": None,
                }
                if prev_node is not None and prev_node["has_ca"] and has_ca:
                    if prev_node["has_mc"] and has_mc:
                        p_c = prev_node["c"]
                        c_n = node["n"]
                        dx, dy, dz = p_c[0] - c_n[0], p_c[1] - c_n[1], p_c[2] - c_n[2]
                        linked = (dx * dx + dy * dy + dz * dz) < 4.0
                    else:
                        p_ca = prev_node["ca"]
                        c_ca = node["ca"]
                        dx, dy, dz = p_ca[0] - c_ca[0], p_ca[1] - c_ca[1], p_ca[2] - c_ca[2]
                        linked = (dx * dx + dy * dy + dz * dz) < 20.25
                    if linked:
                        prev_node["next"] = node
                        node["prev"] = prev_node
                prev_node = node
                if sid not in cur_conf_res:
                    cur_conf_res[sid] = node

        for cname in conformer_ids:
            for sid, node in results_by_conformer[cname].items():
                p1 = node["prev"]
                n1 = node["next"]
                has_ca3 = node["has_ca"] and (p1 is not None and p1["has_ca"]) and (n1 is not None and n1["has_ca"])
                has_mu_in = has_ca3 and (p1["prev"] is not None and p1["prev"]["has_ca"])
                has_mu_out = has_ca3 and (n1["next"] is not None and n1["next"]["has_ca"])
                has_nu = node["has_mc"] and (p1 is not None and p1["has_mc"]) and (n1 is not None and n1["has_mc"])
                has_om = node["has_mc"] and (p1 is not None and p1["has_mc"])
                if not (has_ca3 or has_nu or has_om):
                    continue
                geom_nodes.append(node)
                ca0_l.append(p1["prev"]["ca"] if has_mu_in else u1)
                ca1_l.append(p1["ca"] if (has_ca3 or has_nu or has_om) else z0)
                ca2_l.append(node["ca"] if (has_ca3 or has_nu or has_om) else u2)
                ca3_l.append(n1["ca"] if (has_ca3 or has_nu) else u3)
                ca4_l.append(n1["next"]["ca"] if has_mu_out else u1)
                o1_l.append(p1["o"] if has_nu else u1)
                o2_l.append(node["o"] if has_nu else u3)
                c1_l.append(p1["c"] if has_om else u1)
                n2_l.append(node["n"] if has_om else u3)
                has_ca3_l.append(has_ca3)
                has_mu_in_l.append(has_mu_in)
                has_mu_out_l.append(has_mu_out)
                has_nu_l.append(has_nu)
                has_om_l.append(has_om)

        pure_owned_lists.append(("alt", len(chains_work)))
        chains_work.append((conformer_ids, results_by_conformer, all_keys))

    if pure_eval_targets:
        n_f = len(pure_eval_targets)
        b_f = _next_bucket(n_f)
        cat_pad = lambda chunks, dt: jnp.asarray(_pad_to(np.concatenate(chunks).astype(dt, copy=False), b_f))
        cab_f, ca_f, vnu_f = _cablam_fused_batch_kernel(
            cat_pad(f_ca0, np.float64),
            cat_pad(f_ca1, np.float64),
            cat_pad(f_ca2, np.float64),
            cat_pad(f_ca3, np.float64),
            cat_pad(f_ca4, np.float64),
            cat_pad(f_o1, np.float64),
            cat_pad(f_o2, np.float64),
            cat_pad(f_c1, np.float64),
            cat_pad(f_n2, np.float64),
            cat_pad(f_has_nu, np.bool_),
            cat_pad(f_has_om, np.bool_),
            cat_pad(f_tidx, np.int32),
            cab_tables["pep_tables"],
            cab_tables["ca_tables"],
        )
        cab_f_list = np.asarray(cab_f[:n_f], dtype=np.float64).tolist()
        ca_f_list = np.asarray(ca_f[:n_f], dtype=np.float64).tolist()
        vnu_f_list = np.asarray(vnu_f[:n_f], dtype=bool).tolist()
        for rec, cb_s, ca_s, v_nu in zip(pure_eval_targets, cab_f_list, ca_f_list, vnu_f_list):
            if v_nu:
                rec[4] = cb_s
            rec[5] = ca_s

    alt_owned_by_idx = {}
    if geom_nodes:
        n_g = len(geom_nodes)
        b_g = _next_bucket(n_g)
        pad = lambda lst, dt: jnp.asarray(_pad_to(np.asarray(lst, dtype=dt), b_g))
        mu_in_a, mu_out_a, nu_a, ca_v_a, om_a, val_nu_a = _cablam_geom_batch_kernel(
            pad(ca0_l, np.float64),
            pad(ca1_l, np.float64),
            pad(ca2_l, np.float64),
            pad(ca3_l, np.float64),
            pad(ca4_l, np.float64),
            pad(o1_l, np.float64),
            pad(o2_l, np.float64),
            pad(c1_l, np.float64),
            pad(n2_l, np.float64),
            pad(has_ca3_l, np.bool_),
            pad(has_mu_in_l, np.bool_),
            pad(has_mu_out_l, np.bool_),
            pad(has_nu_l, np.bool_),
            pad(has_om_l, np.bool_),
        )
        mu_in_np = np.asarray(mu_in_a[:n_g], dtype=np.float64)
        mu_out_np = np.asarray(mu_out_a[:n_g], dtype=np.float64)
        nu_np = np.asarray(nu_a[:n_g], dtype=np.float64)
        ca_v_np = np.asarray(ca_v_a[:n_g], dtype=np.float64)
        om_np = np.asarray(om_a[:n_g], dtype=np.float64)
        val_nu_np = np.asarray(val_nu_a[:n_g], dtype=bool)

        for i, node in enumerate(geom_nodes):
            if has_ca3_l[i]:
                node["ca_virt"] = float(ca_v_np[i])
            if has_mu_in_l[i]:
                node["mu_in"] = float(mu_in_np[i])
            if has_mu_out_l[i]:
                node["mu_out"] = float(mu_out_np[i])
            if val_nu_np[i]:
                node["nu"] = float(nu_np[i])
            if has_om_l[i]:
                node["omega"] = float(om_np[i])

    if chains_work:
        eval_nodes = []
        ev_mu_in, ev_mu_out, ev_nu, ev_ca_v, ev_tidx = [], [], [], [], []
        for cw_idx, (conformer_ids, results_by_conformer, all_keys) in enumerate(chains_work):
            chain_results = []
            for sid in all_keys:
                res_node = None
                found_idx = 0
                for idx, cname in enumerate(conformer_ids):
                    if sid in results_by_conformer[cname]:
                        res_node = results_by_conformer[cname][sid]
                        found_idx = idx
                        break
                found_meaningful_alt = False
                m_base = (res_node["mu_in"], res_node["mu_out"], res_node["nu"])
                for other_cname in conformer_ids[found_idx + 1 :]:
                    if sid in results_by_conformer[other_cname]:
                        other_node = results_by_conformer[other_cname][sid]
                        m_other = (other_node["mu_in"], other_node["mu_out"], other_node["nu"])
                        if m_base != m_other:
                            chain_results.append(other_node)
                            found_meaningful_alt = True
                if not found_meaningful_alt:
                    res_node["altloc"] = ""
                chain_results.append(res_node)

            chain_results.sort(
                key=lambda r: (r["chain_id"].rjust(2), _hy36decode(r["resseq"]), r["icode"], r["altloc"])
            )
            cw_owned = []
            for r in chain_results:
                key = (r["chain_id"].strip(), residue_id(r["resseq"], r["icode"]))
                if r["has_ca"] and owned(*key):
                    r["key"] = key
                    r["cablam_score"] = None
                    r["ca_geom_score"] = None
                    cw_owned.append(r)
                    rn_up = r["resname"].strip().upper()
                    if rn_up in common_aa and r["mu_in"] is not None and r["mu_out"] is not None:
                        if r["nu"] is not None or r["ca_virt"] is not None:
                            if rn_up == "GLY":
                                t_i = 1
                            elif rn_up == "PRO":
                                om = r["omega"]
                                t_i = 3 if (om is not None and -90.0 <= om <= 90.0) else 2
                            else:
                                t_i = 0
                            eval_nodes.append(r)
                            ev_mu_in.append(r["mu_in"])
                            ev_mu_out.append(r["mu_out"])
                            ev_nu.append(r["nu"] if r["nu"] is not None else 0.0)
                            ev_ca_v.append(r["ca_virt"] if r["ca_virt"] is not None else 0.0)
                            ev_tidx.append(t_i)
            alt_owned_by_idx[cw_idx] = cw_owned

        if eval_nodes:
            n_e = len(eval_nodes)
            b_e = _next_bucket(n_e)
            pad = lambda lst, dt: jnp.asarray(_pad_to(np.asarray(lst, dtype=dt), b_e))
            cab_a, ca_a = _cablam_eval_batch_kernel(
                pad(ev_mu_in, np.float64),
                pad(ev_mu_out, np.float64),
                pad(ev_nu, np.float64),
                pad(ev_ca_v, np.float64),
                pad(ev_tidx, np.int32),
                cab_tables["pep_tables"],
                cab_tables["ca_tables"],
            )
            cab_np = np.asarray(cab_a[:n_e], dtype=np.float64)
            ca_np = np.asarray(ca_a[:n_e], dtype=np.float64)
            for i, r in enumerate(eval_nodes):
                if r["nu"] is not None:
                    r["cablam_score"] = float(cab_np[i])
                if r["ca_virt"] is not None:
                    r["ca_geom_score"] = float(ca_np[i])

    for kind, payload in pure_owned_lists:
        if kind == "pure":
            records.extend(tuple(rec) for rec in payload)
        else:
            for r in alt_owned_by_idx.get(payload, ()):
                records.append(
                    r["key"]
                    + (
                        r["resname"].strip(),
                        r["altloc"].strip(),
                        r["cablam_score"],
                        r["ca_geom_score"],
                    )
                )
    return records, classes

