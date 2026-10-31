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
        return prepare_noh_hierarchy(read_hierarchy(h_or_path))
    return h_or_path


def ramachandran(hierarchy):
    """
    Every ramalyze result as (chain, resid, resname, altloc, class, phi, psi, score).
    Accepts a JaxHierarchy, an iotbx.pdb.hierarchy.root, or a path to a PDB/mmCIF file.
    Evaluates all candidate peptide triples across the entire model in a single
    vectorized JAX/GPU batch while preserving exact per-chain deduplication and sort order.
    """
    h = _ensure_hierarchy(hierarchy)
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

    # Gather all valid triples across all chains
    triples_meta = []
    ca0_list, c0_list, n1_list, ca1_list, c1_list, n2_list = [], [], [], [], [], []
    om_ca0_list, om_c0_list, om_n1_list, om_ca1_list = [], [], [], []
    is_gly_list, is_pro_list, is_ile_val_list, next_is_pro_list = [], [], [], []

    for cid in chain_ids_ordered:
        uniqueness_keys = set()
        for ch in chains_by_cid[cid]:
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
                    # Check phi/psi atoms exist
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

                    c0_list.append(c0.xyz)
                    n1_list.append(n1.xyz)
                    ca1_list.append(ca1.xyz)
                    c1_list.append(c1.xyz)
                    n2_list.append(n2.xyz)
                    ca0_list.append(ca0_xyz)
                    om_ca0_list.append(ca0_xyz)
                    om_c0_list.append(c0.xyz if has_omega else (1.0, 0.0, 0.0))
                    om_n1_list.append(n1.xyz if has_omega else (0.0, 1.0, 0.0))
                    om_ca1_list.append(ca1.xyz if has_omega else (0.0, 0.0, 1.0))

                    is_gly_list.append(rn1 == "GLY")
                    is_pro_list.append(rn1 == "PRO")
                    is_ile_val_list.append(rn1 in ("ILE", "VAL"))
                    next_is_pro_list.append(rn2 == "PRO")

                    triples_meta.append((
                        cid,
                        ch.id,
                        r1.resseq,
                        r1.icode,
                        r1.resseq_as_int(),
                        rn1,
                        altloc,
                    ))

    if not triples_meta:
        return []

    n_triples = len(triples_meta)
    bucket_n = _next_bucket(n_triples)
    pad = lambda lst, dt: jnp.asarray(_pad_to(np.asarray(lst, dtype=dt), bucket_n))

    phi_arr, psi_arr, score_arr, _, eval_arr = rama_eval_batch_jax(
        pad(ca0_list, np.float64),
        pad(c0_list, np.float64),
        pad(n1_list, np.float64),
        pad(ca1_list, np.float64),
        pad(c1_list, np.float64),
        pad(n2_list, np.float64),
        pad(om_ca0_list, np.float64),
        pad(om_c0_list, np.float64),
        pad(om_n1_list, np.float64),
        pad(om_ca1_list, np.float64),
        pad(is_gly_list, np.bool_),
        pad(is_pro_list, np.bool_),
        pad(is_ile_val_list, np.bool_),
        pad(next_is_pro_list, np.bool_),
        tables["rama_tables"],
    )

    phi_np = np.asarray(phi_arr[:n_triples], dtype=np.float64)
    psi_np = np.asarray(psi_arr[:n_triples], dtype=np.float64)
    score_np = np.asarray(score_arr[:n_triples], dtype=np.float64)
    eval_np = np.asarray(eval_arr[:n_triples], dtype=np.int32)
    eval_names = ("OUTLIER", "Allowed", "Favored")

    # Group by cid, sort each cid's results by (chain_id, resseq_as_int, icode, altloc)
    # matching ramalyze's sorted(self.results, key=lambda r: r.sort_key())
    by_cid = {cid: [] for cid in chain_ids_ordered}
    for i, (cid, ch_id, resseq, icode, resseq_int, rn1, altloc) in enumerate(triples_meta):
        sort_key = (ch_id, resseq_int, icode, altloc)
        rec = (
            ch_id.strip(),
            residue_id(resseq, icode),
            rn1,
            altloc.strip(),
            eval_names[int(eval_np[i])],
            float(phi_np[i]),
            float(psi_np[i]),
            float(score_np[i]),
        )
        by_cid[cid].append((sort_key, i, rec))

    out = []
    for cid in chain_ids_ordered:
        items = by_cid[cid]
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
    tables = get_tables()
    aa_to_idx = tables["aa_to_idx"]
    chi_atoms_map = tables["chi_atoms"]
    box_names = tables["box_names"]
    eval_names = ("OUTLIER", "Allowed", "Favored")

    model = h.models()[0]
    meta = []
    p0_list, p1_list, p2_list, p3_list = [], [], [], []
    chi_valid_list = []
    aa_idx_list = []
    is_sym_list = []

    zero_pt = (0.0, 0.0, 0.0)
    unit_pts = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    for ch in model.chains():
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
                # Standardise lookup by stripped atom name (matching sidechain_angles.measureChiAngles)
                by_name = {k.strip(): v for k, v in atom_dict.items()}
                chi_defs = chi_atoms_map[resname_lower]

                # Check all required chi atoms exist (including chi1..3 for PRO!)
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

                p0_list.append(c_p0)
                p1_list.append(c_p1)
                p2_list.append(c_p2)
                p3_list.append(c_p3)
                chi_valid_list.append(c_val)
                aa_idx_list.append(aa_i)
                is_sym_list.append(resname_lower in SYM_AA)
                meta.append((cid_strip, resid_str, resname_strip, ag.altloc.strip(), aa_i))

    if not meta:
        return []

    n_rota = len(meta)
    bucket_n = _next_bucket(n_rota)
    pad = lambda lst, dt: jnp.asarray(_pad_to(np.asarray(lst, dtype=dt), bucket_n))

    score_arr, eval_arr, box_arr = rotamer_eval_batch_jax(
        pad(p0_list, np.float64),
        pad(p1_list, np.float64),
        pad(p2_list, np.float64),
        pad(p3_list, np.float64),
        pad(chi_valid_list, np.bool_),
        pad(aa_idx_list, np.int32),
        pad(is_sym_list, np.bool_),
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

    score_np = np.asarray(score_arr[:n_rota], dtype=np.float64)
    eval_np = np.asarray(eval_arr[:n_rota], dtype=np.int32)
    box_np = np.asarray(box_arr[:n_rota], dtype=np.int32)

    out = []
    for i, (cid_strip, resid_str, resname_strip, alt_strip, aa_i) in enumerate(meta):
        ev_code = int(eval_np[i])
        ev_str = eval_names[ev_code]
        if ev_code == 0:
            rname = "OUTLIER"
        else:
            b_idx = int(box_np[i])
            rname = box_names[aa_i][b_idx] if b_idx >= 0 else "UNCLASSIFIED"
        out.append((
            cid_strip,
            resid_str,
            resname_strip,
            alt_strip,
            ev_str,
            rname,
            float(score_np[i]),
        ))
    return out
