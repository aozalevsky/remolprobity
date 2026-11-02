"""
GPU-accelerated covalent bond and angle geometry validation (v34 CDL + v33)
in JAX (float64), matching remolprobity.geometry.score.
"""
import functools
import json
import os

import numpy as np

from .ids import atom_tuple, residue_key
from .io import InputError
from .jax_residues import _next_bucket, dihedral_angles_jax
from .jax_tables import (
    _get_jax,
    build_cdl_npz,
    call_gpu,
    default_cdl_path,
    lazy_jit,
    register_gpu_op,
)

DEFINITIONS = ('v34', 'v33')
V34_CUTOFF = 5.0
V33_CUTOFF = 4.0

# Origin IDs in cctbx.geometry_restraints.linking_class / auto_linking_types:
# 0: 'covalent geometry'
# 1: 'SS BOND'
# 2: 'hydrogen bonds'
# 3: 'metal coordination'
# 4: 'edits'
V34_SCOREABLE_ORIGIN_IDS = (0, 1)
V33_COVALENT_ORIGIN_IDS = (
    0, 1, 3, 4, 5, 10, 11, 12,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86,
)
V33_OUTLIER_ORIGIN_ID = 0
EDITS_ORIGIN_ID = 4

DEFAULT_CDL_NPZ_PATH = default_cdl_path()

CDL_COLUMNS = (
    "", "",
    "mCNA", "sCNA",
    "mNAB", "sNAB",
    "mNAC", "sNAC",
    "mBAC", "sBAC",
    "mACO", "sACO",
    "mACN", "sACN",
    "mOCN", "sOCN",
    "mCN", "sCN",
    "mNA", "sNA",
    "mAB", "sAB",
    "mAC", "sAC",
    "mCO", "sCO",
    "mCND", "sCND",
    "mAND", "sAND",
    "mNDG", "sNDG",
    "mABG", "sABG",
    "mBGD", "sBGD",
    "mBG", "sBG",
    "mGD", "sGD",
    "mND", "sND",
)


@functools.lru_cache(maxsize=1)
def get_cpu_cdl_tables(npz_path=None):
    """Load the Conformation-Dependent Library (CDL) metadata and NumPy tables without importing JAX."""
    npz_path = npz_path or default_cdl_path()
    if not os.path.isfile(npz_path):
        build_cdl_npz(npz_path, only_if_missing=True)
    data = np.load(npz_path, allow_pickle=False)
    group_names = json.loads(str(data["group_names_json"]))
    before_pro_groups = json.loads(str(data["before_pro_groups_json"]))
    not_before_pro_groups = json.loads(str(data["not_before_pro_groups_json"]))
    parent_child = json.loads(str(data["parent_child_json"]))
    three_letter_given_one_letter = json.loads(str(data["three_letter_given_one_letter_json"]))
    pro_children = {c for c, p in parent_child.items() if p == "P"}
    return {
        "group_names": group_names,
        "group_to_idx": {g: i for i, g in enumerate(group_names)},
        "cdl_grid_np": np.asarray(data["cdl_grid"], dtype=np.float64),
        "cdl_stat_type": np.asarray(data["cdl_stat_type"]),
        "cis_pro_eh99": np.asarray(data["cis_pro_eh99"], dtype=np.float64),
        "cis_nonpro": np.asarray(data["cis_nonpro"], dtype=np.float64),
        "before_pro_groups": before_pro_groups,
        "not_before_pro_groups": not_before_pro_groups,
        "parent_child": parent_child,
        "three_letter_given_one_letter": three_letter_given_one_letter,
        "pro_children": pro_children,
    }


@functools.lru_cache(maxsize=1)
def get_cdl_tables(npz_path=None):
    """Load the Conformation-Dependent Library (CDL) tables onto GPU."""
    _, jnp_mod = _get_jax()
    cpu = get_cpu_cdl_tables(npz_path)
    return dict(cpu, cdl_grid=jnp_mod.asarray(cpu["cdl_grid_np"], dtype=jnp_mod.float64))


def _bucket_xyz_np(xyz):
    xyz_arr = np.asarray(xyz, dtype=np.float64)
    n_x = len(xyz_arr)
    if n_x == 0:
        return np.zeros((64, 3), dtype=np.float64)
    b_x = _next_bucket(n_x)
    if b_x > n_x:
        xyz_arr = np.pad(xyz_arr, ((0, b_x - n_x), (0, 0)), mode="constant")
    return xyz_arr


def _bucket_xyz_jax(xyz):
    jax_mod, jnp_mod = _get_jax()
    if isinstance(xyz, jax_mod.Array):
        return xyz
    return jnp_mod.asarray(_bucket_xyz_np(xyz), dtype=jnp_mod.float64)


@lazy_jit()
def _eval_bonds_kernel(xyz, iseqs, ideal, weight):
    """
    Evaluate bond lengths, deltas, sigmas, and Z-scores in float64 on GPU.
    Matches cctbx::geometry_restraints::bond.
    """
    p0 = xyz[iseqs[:, 0]]
    p1 = xyz[iseqs[:, 1]]
    diff = p0 - p1
    model_val = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    delta = ideal - model_val
    sigma = jnp.sqrt(1.0 / weight)
    z = -delta / sigma
    return model_val, delta, sigma, z


@register_gpu_op("eval_bonds")
def _eval_bonds_op(xyz_p, iseqs_p, ideal_p, weight_p):
    _, jnp_mod = _get_jax()
    m, d, s, z = _eval_bonds_kernel(
        _bucket_xyz_jax(xyz_p),
        jnp_mod.asarray(iseqs_p, dtype=jnp_mod.int32),
        jnp_mod.asarray(ideal_p, dtype=jnp_mod.float64),
        jnp_mod.asarray(weight_p, dtype=jnp_mod.float64),
    )
    return np.asarray(m), np.asarray(d), np.asarray(s), np.asarray(z)


def eval_bonds_jax(xyz, iseqs, ideal, weight):
    """Bucket-padded wrapper around _eval_bonds_kernel."""
    n = len(iseqs)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    bucket = _next_bucket(n)
    xyz_p = _bucket_xyz_np(xyz)
    if bucket > n:
        pad = bucket - n
        iseqs_p = np.pad(iseqs, ((0, pad), (0, 0)), mode="constant")
        ideal_p = np.pad(ideal, (0, pad), mode="constant")
        weight_p = np.pad(weight, (0, pad), mode="constant", constant_values=1.0)
    else:
        iseqs_p, ideal_p, weight_p = iseqs, ideal, weight
    m, d, s, z = call_gpu("eval_bonds", xyz_p, iseqs_p, ideal_p, weight_p)
    return m[:n], d[:n], s[:n], z[:n]


@lazy_jit()
def _eval_angles_kernel(xyz, iseqs, ideal, weight):
    """
    Evaluate 3-point bond angles (in degrees), wrapped deltas, sigmas, and Z-scores
    in float64 on GPU. Matches cctbx::geometry_restraints::angle and angle_delta_deg.
    """
    p0 = xyz[iseqs[:, 0]]
    p1 = xyz[iseqs[:, 1]]
    p2 = xyz[iseqs[:, 2]]
    d_01 = p0 - p1
    d_21 = p2 - p1
    d_01_abs = jnp.sqrt(jnp.sum(d_01 * d_01, axis=-1))
    d_21_abs = jnp.sqrt(jnp.sum(d_21 * d_21, axis=-1))
    have_angle = (d_01_abs > 0.0) & (d_21_abs > 0.0)

    safe_01 = jnp.where(d_01_abs > 0.0, d_01_abs, 1.0)[..., None]
    safe_21 = jnp.where(d_21_abs > 0.0, d_21_abs, 1.0)[..., None]
    d_01_unit = d_01 / safe_01
    d_21_unit = d_21 / safe_21

    cos_angle = jnp.clip(jnp.sum(d_01_unit * d_21_unit, axis=-1), -1.0, 1.0)
    pi_180 = jnp.pi / 180.0
    angle_model = jnp.arccos(cos_angle) / pi_180

    # cctbx::geometry_restraints::angle_delta_deg(angle_model, angle_ideal, 1)
    raw_d = jnp.fmod(ideal - angle_model, 360.0)
    delta = jnp.where(
        raw_d < -180.0,
        raw_d + 360.0,
        jnp.where(raw_d > 180.0, raw_d - 360.0, raw_d),
    )
    delta = jnp.where(have_angle, delta, 0.0)
    sigma = jnp.sqrt(1.0 / weight)
    z = -delta / sigma
    return angle_model, delta, sigma, z


@register_gpu_op("eval_angles")
def _eval_angles_op(xyz_p, iseqs_p, ideal_p, weight_p):
    _, jnp_mod = _get_jax()
    m, d, s, z = _eval_angles_kernel(
        _bucket_xyz_jax(xyz_p),
        jnp_mod.asarray(iseqs_p, dtype=jnp_mod.int32),
        jnp_mod.asarray(ideal_p, dtype=jnp_mod.float64),
        jnp_mod.asarray(weight_p, dtype=jnp_mod.float64),
    )
    return np.asarray(m), np.asarray(d), np.asarray(s), np.asarray(z)


def eval_angles_jax(xyz, iseqs, ideal, weight):
    """Bucket-padded wrapper around _eval_angles_kernel."""
    n = len(iseqs)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    bucket = _next_bucket(n)
    xyz_p = _bucket_xyz_np(xyz)
    if bucket > n:
        pad = bucket - n
        iseqs_p = np.pad(iseqs, ((0, pad), (0, 0)), mode="constant")
        ideal_p = np.pad(ideal, (0, pad), mode="constant")
        weight_p = np.pad(weight, (0, pad), mode="constant", constant_values=1.0)
    else:
        iseqs_p, ideal_p, weight_p = iseqs, ideal, weight
    m, d, s, z = call_gpu("eval_angles", xyz_p, iseqs_p, ideal_p, weight_p)
    return m[:n], d[:n], s[:n], z[:n]


@lazy_jit()
def _eval_dihedrals_kernel(
    xyz, iseqs, ideal, weight, periodicity, slack, limit, top_out, alt_ideals
):
    """
    Evaluate 4-point dihedral restraints (in degrees), residuals, and sigma scores
    in float64 on GPU. Matches cctbx::geometry_restraints::dihedral.
    """
    p0 = xyz[iseqs[:, 0]]
    p1 = xyz[iseqs[:, 1]]
    p2 = xyz[iseqs[:, 2]]
    p3 = xyz[iseqs[:, 3]]
    d_01 = p0 - p1
    d_21 = p2 - p1
    d_23 = p2 - p3
    n_0121 = jnp.cross(d_01, d_21)
    n_2123 = jnp.cross(d_21, d_23)
    n_0121_norm = jnp.sum(n_0121 * n_0121, axis=-1)
    n_2123_norm = jnp.sum(n_2123 * n_2123, axis=-1)
    have_angle = (n_0121_norm != 0.0) & (n_2123_norm != 0.0)

    angle_model = dihedral_angles_jax(p0, p1, p2, p3)
    half_period = 180.0 / jnp.maximum(1, jnp.abs(periodicity)).astype(jnp.float64)
    period = 2.0 * half_period
    raw_d = jnp.fmod(ideal - angle_model, period)
    delta = jnp.where(
        raw_d < -half_period,
        raw_d + period,
        jnp.where(raw_d > half_period, raw_d - period, raw_d),
    )

    # Optional alt_angle_ideals (shape (N, 6), NaN for unused)
    for k in range(6):
        ak = alt_ideals[:, k]
        valid_alt = ~jnp.isnan(ak)
        a_val = jnp.where(valid_alt, ak, 0.0)
        a_val = jnp.where(a_val < 0.0, a_val + 360.0, a_val)
        raw_di = jnp.fmod(a_val - angle_model, 360.0)
        delta_i = jnp.where(
            raw_di < -180.0,
            raw_di + 360.0,
            jnp.where(raw_di > 180.0, raw_di - 360.0, raw_di),
        )
        delta = jnp.where(valid_alt & (jnp.abs(delta) > jnp.abs(delta_i)), delta_i, delta)

    delta = jnp.where(have_angle, delta, 0.0)
    delta_local = jnp.where(
        slack <= 0.0,
        delta,
        jnp.where(
            delta > slack,
            delta - slack,
            jnp.where(delta < -slack, delta + slack, 0.0),
        ),
    )

    pi_180 = jnp.pi / 180.0
    p_f = jnp.maximum(1, periodicity).astype(jnp.float64)
    sin_term = (9600.0 / (p_f * p_f)) * (1.0 - jnp.cos(p_f * delta_local * pi_180))
    use_top = (periodicity <= 0) & top_out & (limit >= 0.0)
    top = jnp.where(use_top, jnp.maximum(weight * limit * limit, 1e-30), 1.0)
    top_res = top * (1.0 - jnp.exp(-weight * delta_local * delta_local / top))
    harm_res = weight * delta_local * delta_local
    residual = jnp.where(
        periodicity > 0,
        weight * sin_term,
        jnp.where(use_top, top_res, harm_res),
    )
    sigma = jnp.sqrt(1.0 / weight)
    score = jnp.abs(delta) / sigma
    return angle_model, delta, residual, score


@register_gpu_op("eval_dihedrals")
def _eval_dihedrals_op(
    xyz_p, iseqs_p, ideal_p, weight_p, period_p, slack_p, limit_p, top_p, alt_p
):
    _, jnp_mod = _get_jax()
    m, d, r, sc = _eval_dihedrals_kernel(
        _bucket_xyz_jax(xyz_p),
        jnp_mod.asarray(iseqs_p, dtype=jnp_mod.int32),
        jnp_mod.asarray(ideal_p, dtype=jnp_mod.float64),
        jnp_mod.asarray(weight_p, dtype=jnp_mod.float64),
        jnp_mod.asarray(period_p, dtype=jnp_mod.int32),
        jnp_mod.asarray(slack_p, dtype=jnp_mod.float64),
        jnp_mod.asarray(limit_p, dtype=jnp_mod.float64),
        jnp_mod.asarray(top_p, dtype=jnp_mod.bool_),
        jnp_mod.asarray(alt_p, dtype=jnp_mod.float64),
    )
    return np.asarray(m), np.asarray(d), np.asarray(r), np.asarray(sc)


def eval_dihedrals_jax(
    xyz, iseqs, ideal, weight, periodicity, slack, limit, top_out, alt_ideals
):
    """Bucket-padded wrapper around _eval_dihedrals_kernel."""
    n = len(iseqs)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    bucket = _next_bucket(n)
    xyz_p = _bucket_xyz_np(xyz)
    if bucket > n:
        pad = bucket - n
        iseqs_p = np.pad(iseqs, ((0, pad), (0, 0)), mode="constant")
        ideal_p = np.pad(ideal, (0, pad), mode="constant")
        weight_p = np.pad(weight, (0, pad), mode="constant", constant_values=1.0)
        period_p = np.pad(periodicity, (0, pad), mode="constant")
        slack_p = np.pad(slack, (0, pad), mode="constant")
        limit_p = np.pad(limit, (0, pad), mode="constant", constant_values=-1.0)
        top_p = np.pad(top_out, (0, pad), mode="constant")
        alt_p = np.pad(alt_ideals, ((0, pad), (0, 0)), mode="constant", constant_values=np.nan)
    else:
        iseqs_p, ideal_p, weight_p = iseqs, ideal, weight
        period_p, slack_p, limit_p, top_p, alt_p = periodicity, slack, limit, top_out, alt_ideals
    m, d, r, sc = call_gpu(
        "eval_dihedrals",
        xyz_p, iseqs_p, ideal_p, weight_p, period_p, slack_p, limit_p, top_p, alt_p,
    )
    return m[:n], d[:n], r[:n], sc[:n]


@lazy_jit()
def _eval_chiralities_kernel(xyz, iseqs, ideal, weight, both_signs):
    """
    Evaluate 4-point chiral volumes, deltas, residuals, and sigma scores in float64 on GPU.
    Matches cctbx::geometry_restraints::chirality.
    """
    p0 = xyz[iseqs[:, 0]]
    p1 = xyz[iseqs[:, 1]]
    p2 = xyz[iseqs[:, 2]]
    p3 = xyz[iseqs[:, 3]]
    d_01 = p1 - p0
    d_02 = p2 - p0
    d_03 = p3 - p0
    volume_model = jnp.sum(d_01 * jnp.cross(d_02, d_03), axis=-1)
    delta_sign = jnp.where(both_signs & (volume_model < 0.0), 1.0, -1.0)
    delta = ideal + delta_sign * volume_model
    residual = weight * delta * delta
    sigma = jnp.sqrt(1.0 / weight)
    score = jnp.abs(delta) / sigma
    return volume_model, delta, residual, score


@register_gpu_op("eval_chiralities")
def _eval_chiralities_op(xyz_p, iseqs_p, ideal_p, weight_p, both_p):
    _, jnp_mod = _get_jax()
    m, d, r, sc = _eval_chiralities_kernel(
        _bucket_xyz_jax(xyz_p),
        jnp_mod.asarray(iseqs_p, dtype=jnp_mod.int32),
        jnp_mod.asarray(ideal_p, dtype=jnp_mod.float64),
        jnp_mod.asarray(weight_p, dtype=jnp_mod.float64),
        jnp_mod.asarray(both_p, dtype=jnp_mod.bool_),
    )
    return np.asarray(m), np.asarray(d), np.asarray(r), np.asarray(sc)


def eval_chiralities_jax(xyz, iseqs, ideal, weight, both_signs):
    """Bucket-padded wrapper around _eval_chiralities_kernel."""
    n = len(iseqs)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    bucket = _next_bucket(n)
    xyz_p = _bucket_xyz_np(xyz)
    if bucket > n:
        pad = bucket - n
        iseqs_p = np.pad(iseqs, ((0, pad), (0, 0)), mode="constant")
        ideal_p = np.pad(ideal, (0, pad), mode="constant")
        weight_p = np.pad(weight, (0, pad), mode="constant", constant_values=1.0)
        both_p = np.pad(both_signs, (0, pad), mode="constant")
    else:
        iseqs_p, ideal_p, weight_p, both_p = iseqs, ideal, weight, both_signs
    m, d, r, sc = call_gpu("eval_chiralities", xyz_p, iseqs_p, ideal_p, weight_p, both_p)
    return m[:n], d[:n], r[:n], sc[:n]


def _jacobi_3x3_normal_single(a_init):
    """
    Exact replica of scitbx::matrix::eigensystem::detail::real_symmetric_given_lower_triangle
    (scitbx/matrix/eigensystem.h:22-156) for n=3, returning the eigenvector
    corresponding to the smallest eigenvalue (with bit-exact sign!).
    a_init: (6,) float64 array [m00, m10, m11, m20, m21, m22].
    """
    vecs_init = jnp.eye(3, dtype=jnp.float64)
    anorm = jnp.sqrt(2.0 * (a_init[1] * a_init[1] + a_init[3] * a_init[3] + a_init[4] * a_init[4]))
    anrmx = 1e-10 * anorm / 3.0

    def _rotate_pair(a, vecs, ind, thr, l, m, ll, mm, lm, il, im):
        cond = (a[lm] * a[lm]) > (thr * thr)
        x = 0.5 * (a[ll] - a[mm])
        denom = jnp.sqrt(a[lm] * a[lm] + x * x)
        safe_denom = jnp.where(denom > 0.0, denom, 1.0)
        y = -a[lm] / safe_denom
        y = jnp.where(x < 0.0, -y, y)
        sinx = y / jnp.sqrt(2.0 * (1.0 + jnp.sqrt(jnp.maximum(0.0, 1.0 - y * y))))
        sinx2 = sinx * sinx
        cosx = jnp.sqrt(jnp.maximum(0.0, 1.0 - sinx2))
        cosx2 = cosx * cosx
        sincs = sinx * cosx

        new_il = a[il] * cosx - a[im] * sinx
        new_im = a[il] * sinx + a[im] * cosx
        x2 = 2.0 * a[lm] * sincs
        new_ll = a[ll] * cosx2 + a[mm] * sinx2 - x2
        new_mm = a[ll] * sinx2 + a[mm] * cosx2 + x2
        new_lm = (a[ll] - a[mm]) * sincs + a[lm] * (cosx2 - sinx2)

        a_new = a.at[il].set(new_il).at[im].set(new_im).at[ll].set(new_ll).at[mm].set(new_mm).at[lm].set(new_lm)
        vl = vecs[l] * cosx - vecs[m] * sinx
        vm = vecs[l] * sinx + vecs[m] * cosx
        vecs_new = vecs.at[l].set(vl).at[m].set(vm)

        return (
            jnp.where(cond, a_new, a),
            jnp.where(cond, vecs_new, vecs),
            jnp.where(cond, jnp.int32(1), ind),
        )

    def _inner_cond(state):
        _, _, ind, _ = state
        return ind == 1

    def _inner_body(state):
        a, vecs, _, thr = state
        ind = jnp.int32(0)
        # (l=0, m=1): ll=0, mm=2, lm=1, other i=2 -> il=3, im=4
        a, vecs, ind = _rotate_pair(a, vecs, ind, thr, 0, 1, 0, 2, 1, 3, 4)
        # (l=0, m=2): ll=0, mm=5, lm=3, other i=1 -> il=1, im=4
        a, vecs, ind = _rotate_pair(a, vecs, ind, thr, 0, 2, 0, 5, 3, 1, 4)
        # (l=1, m=2): ll=2, mm=5, lm=4, other i=0 -> il=1, im=3
        a, vecs, ind = _rotate_pair(a, vecs, ind, thr, 1, 2, 2, 5, 4, 1, 3)
        return (a, vecs, ind, thr)

    def _outer_cond(state):
        _, _, thr = state
        return (anorm > 0.0) & (thr > anrmx)

    def _outer_body(state):
        a, vecs, thr = state
        thr_next = thr / 3.0
        a, vecs, _, _ = jax.lax.while_loop(_inner_cond, _inner_body, (a, vecs, jnp.int32(1), thr_next))
        return (a, vecs, thr_next)

    a, vecs, _ = jax.lax.while_loop(_outer_cond, _outer_body, (a_init, vecs_init, anorm))

    # Sort eigenvalues [a[0], a[2], a[5]] and eigenvectors in descending order
    # Matching scitbx/matrix/eigensystem.h:124-150
    e0, e1, e2 = a[0], a[2], a[5]
    v0, v1, v2 = vecs[0], vecs[1], vecs[2]

    # i = 0
    swap01 = e1 > e0
    am0 = jnp.where(swap01, e1, e0)
    im0 = jnp.where(swap01, 1, 0)
    swap02 = e2 > am0
    im0 = jnp.where(swap02, 2, im0)

    e0_1 = jnp.where(im0 == 1, e1, jnp.where(im0 == 2, e2, e0))
    e1_1 = jnp.where(im0 == 1, e0, e1)
    e2_1 = jnp.where(im0 == 2, e0, e2)
    v0_1 = jnp.where(im0 == 1, v1, jnp.where(im0 == 2, v2, v0))
    v1_1 = jnp.where(im0 == 1, v0, v1)
    v2_1 = jnp.where(im0 == 2, v0, v2)

    # i = 1
    swap12 = e2_1 > e1_1
    v2_final = jnp.where(swap12, v1_1, v2_1)
    return v2_final


@lazy_jit()
def _eval_planarities_kernel(xyz, iseqs, weights):
    """
    Evaluate multi-atom planarity restraints on GPU in float64.
    Matches cctbx::geometry_restraints::planarity and mmtbx.validation.restraints.planarities.
    """
    valid = iseqs >= 0
    safe_iseqs = jnp.maximum(iseqs, 0)
    sites = xyz[safe_iseqs]  # (N, M, 3)
    w = jnp.where(valid, weights, 0.0)  # (N, M)
    sum_w = jnp.sum(w, axis=1, keepdims=True)  # (N, 1)
    com = jnp.sum(w[:, :, None] * sites, axis=1) / jnp.maximum(sum_w, 1e-30)  # (N, 3)
    x = jnp.where(valid[:, :, None], sites - com[:, None, :], 0.0)  # (N, M, 3)

    a0 = jnp.sum(w * x[:, :, 0] * x[:, :, 0], axis=1)
    a1 = jnp.sum(w * x[:, :, 1] * x[:, :, 0], axis=1)
    a2 = jnp.sum(w * x[:, :, 1] * x[:, :, 1], axis=1)
    a3 = jnp.sum(w * x[:, :, 2] * x[:, :, 0], axis=1)
    a4 = jnp.sum(w * x[:, :, 2] * x[:, :, 1], axis=1)
    a5 = jnp.sum(w * x[:, :, 2] * x[:, :, 2], axis=1)
    a_mat = jnp.stack([a0, a1, a2, a3, a4, a5], axis=-1)  # (N, 6)

    n_min = jax.vmap(_jacobi_3x3_normal_single)(a_mat)  # (N, 3)
    deltas = jnp.sum(x * n_min[:, None, :], axis=-1)  # (N, M)
    deltas = jnp.where(valid, deltas, 0.0)

    n_sites = jnp.maximum(1, jnp.sum(valid.astype(jnp.int32), axis=1)).astype(jnp.float64)
    rms_deltas = jnp.sqrt(jnp.sum(deltas * deltas, axis=1) / n_sites)
    residual = jnp.sum(w * deltas * deltas, axis=1)

    sigma_site = jnp.sqrt(1.0 / jnp.where(valid & (w > 0.0), w, 1.0))
    # Note: mmtbx/validation/restraints.py:665-672 uses signed a[1] (delta)
    delta_max = jnp.max(jnp.where(valid, deltas, -1e30), axis=1)
    score = jnp.max(jnp.where(valid, deltas / sigma_site, -1e30), axis=1)
    return rms_deltas, delta_max, residual, score


@register_gpu_op("eval_planarities")
def _eval_planarities_op(xyz_p, iseqs_p, weights_p):
    _, jnp_mod = _get_jax()
    rms, dmax, res, sc = _eval_planarities_kernel(
        _bucket_xyz_jax(xyz_p),
        jnp_mod.asarray(iseqs_p, dtype=jnp_mod.int32),
        jnp_mod.asarray(weights_p, dtype=jnp_mod.float64),
    )
    return np.asarray(rms), np.asarray(dmax), np.asarray(res), np.asarray(sc)


def eval_planarities_jax(xyz, iseqs, weights):
    """Bucket-padded wrapper around _eval_planarities_kernel."""
    n = len(iseqs)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    m_cols = iseqs.shape[1]
    # Pad m_cols to next multiple of 8 (or 20) to avoid recompilation across plane sizes
    m_pad = max(20, ((m_cols + 7) // 8) * 8)
    bucket = _next_bucket(n)
    xyz_p = _bucket_xyz_np(xyz)
    pad_n = bucket - n
    pad_m = m_pad - m_cols
    iseqs_p = np.pad(iseqs, ((0, pad_n), (0, pad_m)), mode="constant", constant_values=-1)
    weights_p = np.pad(weights, ((0, pad_n), (0, pad_m)), mode="constant", constant_values=0.0)
    rms, dmax, res, sc = call_gpu("eval_planarities", xyz_p, iseqs_p, weights_p)
    return rms[:n], dmax[:n], res[:n], sc[:n]


@lazy_jit(static_argnums=(3,))
def _accumulate_residue_rmsz_kernel(z_sq, keep_mask, res_idx, n_residues):
    """Accumulate per-residue count and sum(z^2) on GPU via segment_sum."""
    counts = jax.ops.segment_sum(
        keep_mask.astype(jnp.int32), res_idx, num_segments=n_residues
    )
    sums = jax.ops.segment_sum(
        jnp.where(keep_mask, z_sq, 0.0), res_idx, num_segments=n_residues
    )
    return counts, sums


@register_gpu_op("accum_rmsz")
def _accum_rmsz_op(z_sq_p, keep_p, res_idx_p, n_res_bucket):
    _, jnp_mod = _get_jax()
    counts, sums = _accumulate_residue_rmsz_kernel(
        jnp_mod.asarray(z_sq_p, dtype=jnp_mod.float64),
        jnp_mod.asarray(keep_p, dtype=jnp_mod.bool_),
        jnp_mod.asarray(res_idx_p, dtype=jnp_mod.int32),
        int(n_res_bucket),
    )
    return np.asarray(counts), np.asarray(sums)


def _round_to_ten_index(angle_deg):
    """
    Replicate cdl_utils.round_to_ten(d) and map (-180..170) to index 0..35.
    Python's round(x) rounds half to even; in float64 dihedral angles, exact
    half-way multiples of 5.0000000000000000 do not occur except at 0, so
    int(round(float(d) / 10)) * 10 matches.
    """
    t = int(round(float(angle_deg) / 10.0)) * 10
    if t == 180:
        t = -180
    return (t + 180) // 10


@lazy_jit()
def _cdl_lookup_kernel(group_idx, phi_idx, psi_idx, cdl_grid):
    """Gather the 24 CDL target/sigma columns for each (group, phi_bin, psi_bin) on GPU."""
    return cdl_grid[group_idx, phi_idx, psi_idx]


@register_gpu_op("cdl_lookup")
def _cdl_lookup_op(group_indices, phi_indices, psi_indices):
    _, jnp_mod = _get_jax()
    tables = get_cdl_tables()
    res = _cdl_lookup_kernel(
        jnp_mod.asarray(group_indices, dtype=jnp_mod.int32),
        jnp_mod.asarray(phi_indices, dtype=jnp_mod.int32),
        jnp_mod.asarray(psi_indices, dtype=jnp_mod.int32),
        tables["cdl_grid"],
    )
    return np.asarray(res, dtype=np.float64)


def _get_res_type_group(resname1, resname2, tables):
    resname1 = resname1.strip()
    resname2 = resname2.strip()
    if resname2 == "PRO" or resname2 in tables["pro_children"]:
        lookup = tables["before_pro_groups"]
    else:
        lookup = tables["not_before_pro_groups"]
    parent1 = tables["three_letter_given_one_letter"].get(
        tables["parent_child"].get(resname1, "???"), "???"
    )
    for key, res_list in lookup.items():
        if resname1 in res_list or parent1 in res_list:
            return key
    return None


def apply_cdl_updates(
    hierarchy,
    bond_iseqs,
    bond_ideal,
    bond_weight,
    angle_iseqs,
    angle_ideal,
    angle_weight,
    angle_origin_id=None,
    cis_pro_eh99=True,
):
    """
    Apply Conformation-Dependent Library (CDL) updates to base (cdl=False) bond
    and angle restraint arrays using GPU dihedral evaluation and CDL table lookup.
    Matches mmtbx.conformation_dependent_library.update_restraints(..., cis_pro_eh99=True).
    """
    tables = get_cpu_cdl_tables()
    bond_ideal = np.array(bond_ideal, dtype=np.float64, copy=True)
    bond_weight = np.array(bond_weight, dtype=np.float64, copy=True)
    angle_ideal = np.array(angle_ideal, dtype=np.float64, copy=True)
    angle_weight = np.array(angle_weight, dtype=np.float64, copy=True)

    # Build fast lookup maps for bonds (undirected) and angles (directed + reversed)
    bond_map = {}
    for idx, (i0, i1) in enumerate(bond_iseqs):
        i0, i1 = int(i0), int(i1)
        bond_map[(i0, i1)] = idx
        bond_map[(i1, i0)] = idx

    angle_map = {}
    for idx, (i0, i1, i2) in enumerate(angle_iseqs):
        key = (int(i0), int(i1), int(i2))
        angle_map[key] = idx

    def get_angle_idx(names):
        idx = angle_map.get(tuple(names))
        if idx is None:
            idx = angle_map.get((names[2], names[1], names[0]))
        return idx

    # Collect protein threes across conformers, exactly as cdl.generate_protein_threes does
    # Note: cdl.update_restraints selects backbone atoms first:
    # retain_selection="name ca or name c or name n or name o or name cb or name h or name cd or name cg or name HNO"
    # and iterates over chain.conformers(), keeping only common_amino_acid residues.
    from .jax_io import read_hierarchy
    from .jax_tables import COMMON_AMINO_ACIDS

    h = hierarchy if hasattr(hierarchy, "models") else read_hierarchy(hierarchy)
    retain_names = {"CA", "C", "N", "O", "CB", "H", "CD", "CG", "HNO"}

    threes_list = []
    phi_coords = []
    psi_coords = []
    omega1_coords = []

    for chain in h.models()[0].chains():
        for conformer in chain.conformers():
            window = []
            for residue in conformer.residues():
                resname = residue.resname.strip()
                if resname not in COMMON_AMINO_ACIDS:
                    continue
                # Extract retained atoms for this residue
                atom_dict = {}
                for a in residue.atoms():
                    nm = a.name.strip()
                    if nm in retain_names and nm not in atom_dict:
                        atom_dict[nm] = a
                window.append((resname, atom_dict))
                if len(window) > 3:
                    window.pop(0)
                if len(window) >= 2:
                    # Check are_linked() using bond_map
                    while len(window) >= 2:
                        linked = True
                        for k in range(1, len(window)):
                            c_prev = window[k - 1][1].get("C")
                            n_curr = window[k][1].get("N")
                            if c_prev is None or n_curr is None:
                                linked = False
                                break
                            if (c_prev.i_seq, n_curr.i_seq) not in bond_map:
                                linked = False
                                break
                        if linked:
                            break
                        window.pop(0)
                if len(window) == 3:
                    r0, r1, r2 = window
                    # Check phi, psi, omega atoms
                    c0, ca0 = r0[1].get("C"), r0[1].get("CA")
                    n1, ca1, c1 = r1[1].get("N"), r1[1].get("CA"), r1[1].get("C")
                    n2, ca2, c2 = r2[1].get("N"), r2[1].get("CA"), r2[1].get("C")
                    if None in (c0, ca0, n1, ca1, c1, n2, ca2, c2):
                        continue
                    phi_coords.append([c0.xyz, n1.xyz, ca1.xyz, c1.xyz])
                    psi_coords.append([n1.xyz, ca1.xyz, c1.xyz, n2.xyz])
                    omega1_coords.append([ca1.xyz, n1.xyz, c0.xyz, ca0.xyz])
                    threes_list.append((r0, r1, r2))

    if not threes_list:
        return bond_ideal, bond_weight, angle_ideal, angle_weight

    # Batch compute (phi, psi, omega) dihedrals on GPU
    all_quads = np.concatenate(
        [
            np.asarray(phi_coords, dtype=np.float64),
            np.asarray(psi_coords, dtype=np.float64),
            np.asarray(omega1_coords, dtype=np.float64),
        ],
        axis=0,
    )
    n_q = len(all_quads)
    b_q = _next_bucket(n_q)
    if b_q > n_q:
        all_quads_p = np.pad(all_quads, ((0, b_q - n_q), (0, 0), (0, 0)), mode="edge")
    else:
        all_quads_p = all_quads
    all_angles = call_gpu(
        "dihedral_angles",
        all_quads_p[:, 0],
        all_quads_p[:, 1],
        all_quads_p[:, 2],
        all_quads_p[:, 3],
    )[:n_q]
    n_t = len(threes_list)
    phis = all_angles[:n_t]
    psis = all_angles[n_t : 2 * n_t]
    omegas = all_angles[2 * n_t :]

    # Batch lookup CDL grid on GPU
    b_t = _next_bucket(n_t)
    group_indices = np.zeros(b_t, dtype=np.int32)
    phi_indices = np.zeros(b_t, dtype=np.int32)
    psi_indices = np.zeros(b_t, dtype=np.int32)
    valid_group = np.zeros(n_t, dtype=bool)

    for idx, (r0, r1, r2) in enumerate(threes_list):
        g = _get_res_type_group(r1[0], r2[0], tables)
        if g is not None:
            valid_group[idx] = True
            group_indices[idx] = tables["group_to_idx"][g]
            phi_indices[idx] = _round_to_ten_index(phis[idx])
            psi_indices[idx] = _round_to_ten_index(psis[idx])

    cdl_batch = call_gpu("cdl_lookup", group_indices, phi_indices, psi_indices)[:n_t]

    # Replicate RestraintsRegistry and apply_updates
    registry_sum = {}
    registry_n = {}

    def registry_set(key, val):
        if key in registry_sum:
            if registry_sum[key] != val:
                registry_n.setdefault(key, 1)
                registry_n[key] += 1
                registry_sum[key] = registry_sum[key] + val
        else:
            registry_sum[key] = val

    for idx, (r0, r1, r2) in enumerate(threes_list):
        if not valid_group[idx]:
            continue
        is_cis = abs(omegas[idx]) < 45.0
        if is_cis:
            if r1[0] != "PRO" or not cis_pro_eh99:
                continue
            restraint_values = [None] + list(tables["cis_pro_eh99"])
        elif "HNO" in r1[1]:
            continue
        else:
            restraint_values = list(cdl_batch[idx])

        atoms = {}
        for nm in ("C", "CA"):
            if nm in r0[1]:
                atoms[f"{nm}_minus_1"] = r0[1][nm]
        for nm in ("N", "CA", "CB", "C", "O", "H", "CD", "CG"):
            if nm in r1[1]:
                atoms[f"{nm}_i"] = r1[1][nm]
        for nm in ("N", "CA"):
            if nm in r2[1]:
                atoms[f"{nm}_plus_1"] = r2[1][nm]

        for i in range(2, len(restraint_values), 2):
            val = restraint_values[i]
            esd = restraint_values[i + 1]
            if val is None or np.isnan(val):
                continue
            code = CDL_COLUMNS[i][1:]
            if code == "CNA":
                names = ["C_minus_1", "N_i", "CA_i"]
            elif code == "NAB":
                names = ["N_i", "CA_i", "CB_i"]
            elif code == "NAC":
                names = ["N_i", "CA_i", "C_i"]
            elif code == "BAC":
                names = ["CB_i", "CA_i", "C_i"]
            elif code == "ACO":
                names = ["CA_i", "C_i", "O_i"]
            elif code == "ACN":
                names = ["CA_i", "C_i", "N_plus_1"]
            elif code == "OCN":
                names = ["O_i", "C_i", "N_plus_1"]
            elif code == "CN":
                names = ["C_minus_1", "N_i"]
            elif code == "NA":
                names = ["N_i", "CA_i"]
            elif code == "AB":
                names = ["CA_i", "CB_i"]
            elif code == "AC":
                names = ["CA_i", "C_i"]
            elif code == "CO":
                names = ["C_i", "O_i"]
            elif code == "CND":
                names = ["C_minus_1", "N_i", "CD_i"]
            elif code == "AND":
                names = ["CA_i", "N_i", "CD_i"]
            elif code == "NDG":
                names = ["N_i", "CD_i", "CG_i"]
            elif code == "ABG":
                names = ["CA_i", "CB_i", "CG_i"]
            elif code == "BGD":
                names = ["CB_i", "CG_i", "CD_i"]
            elif code == "BG":
                names = ["CB_i", "CG_i"]
            elif code == "GD":
                names = ["CG_i", "CD_i"]
            elif code == "ND":
                names = ["N_i", "CD_i"]
            else:
                continue

            if "CB_i" in names and "CB_i" not in atoms:
                continue
            if "O_i" in names and "O_i" not in atoms:
                continue
            if any(nm not in atoms for nm in names):
                continue
            iseq_list = [atoms[nm].i_seq for nm in names]

            if len(iseq_list) == 3:
                a_idx = get_angle_idx(iseq_list)
                if a_idx is None:
                    continue
                if angle_origin_id is not None and int(angle_origin_id[a_idx]) == EDITS_ORIGIN_ID:
                    continue
                sorted_key = tuple(sorted(iseq_list))
                registry_set(sorted_key, float(val))
                angle_ideal[a_idx] = float(val)
                angle_weight[a_idx] = 1.0 / (float(esd) ** 2)
            elif len(iseq_list) == 2:
                b_idx = bond_map.get((iseq_list[0], iseq_list[1]))
                if b_idx is None:
                    continue
                sorted_key = tuple(sorted(iseq_list))
                registry_set(sorted_key, float(val))
                bond_ideal[b_idx] = float(val)
                bond_weight[b_idx] = 1.0 / (float(esd) ** 2)

        # Planar X-N-H adjustment when H is present
        if "C_minus_1" in atoms and "N_i" in atoms and "CA_i" in atoms and "H_i" in atoms:
            cnca_idx = get_angle_idx([atoms["C_minus_1"].i_seq, atoms["N_i"].i_seq, atoms["CA_i"].i_seq])
            cnh_idx = get_angle_idx([atoms["C_minus_1"].i_seq, atoms["N_i"].i_seq, atoms["H_i"].i_seq])
            canh_idx = get_angle_idx([atoms["CA_i"].i_seq, atoms["N_i"].i_seq, atoms["H_i"].i_seq])
            if cnca_idx is not None and cnh_idx is not None and canh_idx is not None:
                total = angle_ideal[cnca_idx] + angle_ideal[cnh_idx] + angle_ideal[canh_idx]
                diff = (total - 360.0) / 2.0
                angle_ideal[cnh_idx] -= diff
                angle_ideal[canh_idx] -= diff

    # apply_average_updates(registry)
    if registry_n:
        for key in list(registry_n.keys()):
            if len(key) == 2:
                b_idx = bond_map.get(key)
                if b_idx is not None:
                    bond_ideal[b_idx] = registry_sum[key] / registry_n[key]
            elif len(key) == 3:
                rkey = (key[2], key[1], key[0])
                registry_n[rkey] = registry_n[key]
        for a_idx, (i0, i1, i2) in enumerate(angle_iseqs):
            akey = (int(i0), int(i1), int(i2))
            if akey in registry_n:
                angle_ideal[a_idx] = registry_sum[akey] / registry_n[akey]

    return bond_ideal, bond_weight, angle_ideal, angle_weight


_MON_LIB_CACHE = {}
_COMP_CACHE = {}
_NA_COMP_CACHE = {}
_LINK_CACHE = {}
_MOD_CACHE = {}
_NA_ALIASES = {'O1P': 'OP1', 'O2P': 'OP2', 'O3P': 'O3T', 'OP3': 'O3T'}
_NON_COVALENT_SOLVENT_IONS = frozenset((
    'HOH', 'WAT', 'DOD', 'MG', 'ZN', 'NA', 'K', 'CA', 'CL', 'MN', 'FE', 'FE2',
    'CU', 'CO', 'NI', 'CD', 'HG', 'SR', 'BA', 'CS', 'RB', 'LI', 'IOD', 'BR', 'F',
))


def _get_mon_lib():
    if 'srv' not in _MON_LIB_CACHE:
        from mmtbx.monomer_library import server
        from mmtbx.monomer_library.pdb_interpretation import master_params
        from mmtbx.monomer_library.rna_sugar_pucker_analysis import master_phil as rna_phil
        _MON_LIB_CACHE['srv'] = server.server()
        _MON_LIB_CACHE['rna_params'] = rna_phil.extract()
        _MON_LIB_CACHE['interp_params'] = master_params.extract()
    return _MON_LIB_CACHE['srv'], _MON_LIB_CACHE['rna_params'], _MON_LIB_CACHE['interp_params']


def _norm_na_atom(nm):
    s = nm.strip().replace('*', "'")
    return _NA_ALIASES.get(s, s)


def _get_comp_template(resname, has_oxt=False):
    key = (resname, bool(has_oxt))
    if key not in _COMP_CACHE:
        srv, _, _ = _get_mon_lib()
        comp = srv.get_comp_comp_id_direct(resname)
        if comp is None:
            _COMP_CACHE[key] = None
            return None
        if has_oxt:
            m = _get_mod_template('COO')
            if m is not None:
                comp = comp.apply_mod(m)
        valid_atoms = frozenset(a.atom_id.strip() for a in comp.atom_list)
        bonds = []
        for b in comp.bond_list:
            if b.value_dist is None or b.value_dist_esd in (None, 0):
                continue
            bonds.append((b.atom_id_1.strip(), b.atom_id_2.strip(), float(b.value_dist), 1.0 / (float(b.value_dist_esd) ** 2)))
        angles = []
        for a in comp.angle_list:
            if a.value_angle is None or a.value_angle_esd in (None, 0):
                continue
            angles.append((a.atom_id_1.strip(), a.atom_id_2.strip(), a.atom_id_3.strip(), float(a.value_angle), 1.0 / (float(a.value_angle_esd) ** 2)))
        _COMP_CACHE[key] = (comp, bonds, angles, valid_atoms)
    return _COMP_CACHE[key]


def _get_mod_template(mod_id):
    if mod_id not in _MOD_CACHE:
        srv, _, _ = _get_mon_lib()
        mod = srv.mod_mod_id_dict.get(mod_id)
        if mod is None:
            _MOD_CACHE[mod_id] = None
            return None
        _MOD_CACHE[mod_id] = mod
    return _MOD_CACHE[mod_id]


def _get_na_template(resname, pucker_mod, term_mod):
    key = (resname, pucker_mod, term_mod)
    if key not in _NA_COMP_CACHE:
        srv, _, _ = _get_mon_lib()
        comp = srv.get_comp_comp_id_direct(resname)
        if comp is None:
            _NA_COMP_CACHE[key] = None
            return None
        if pucker_mod is not None:
            comp, _ = srv.get_comp_comp_id_mod(comp_comp_id=comp, mod_ids=(pucker_mod,))
        if term_mod is not None:
            m = _get_mod_template(term_mod)
            if m is not None:
                comp = comp.apply_mod(m)
        valid_atoms = frozenset(_norm_na_atom(a.atom_id) for a in comp.atom_list)
        bonds = []
        for b in comp.bond_list:
            if b.value_dist is None or b.value_dist_esd in (None, 0):
                continue
            bonds.append((_norm_na_atom(b.atom_id_1), _norm_na_atom(b.atom_id_2), float(b.value_dist), 1.0 / (float(b.value_dist_esd) ** 2)))
        angles = []
        for a in comp.angle_list:
            if a.value_angle is None or a.value_angle_esd in (None, 0):
                continue
            angles.append((_norm_na_atom(a.atom_id_1), _norm_na_atom(a.atom_id_2), _norm_na_atom(a.atom_id_3), float(a.value_angle), 1.0 / (float(a.value_angle_esd) ** 2)))
        _NA_COMP_CACHE[key] = (comp, bonds, angles, valid_atoms)
    return _NA_COMP_CACHE[key]


def _get_link_template(link_id, norm_na=False):
    key = (link_id, bool(norm_na))
    if key not in _LINK_CACHE:
        srv, _, _ = _get_mon_lib()
        link = srv.link_link_id_dict.get(link_id)
        if link is None:
            _LINK_CACHE[key] = None
            return None
        fn = _norm_na_atom if norm_na else str.strip
        bonds = []
        for b in link.bond_list:
            if b.value_dist is None or b.value_dist_esd in (None, 0):
                continue
            bonds.append((int(b.atom_1_comp_id), fn(b.atom_id_1), int(b.atom_2_comp_id), fn(b.atom_id_2), float(b.value_dist), 1.0 / (float(b.value_dist_esd) ** 2)))
        angles = []
        for a in link.angle_list:
            if a.value_angle is None or a.value_angle_esd in (None, 0):
                continue
            angles.append((int(a.atom_1_comp_id), fn(a.atom_id_1), int(a.atom_2_comp_id), fn(a.atom_id_2), int(a.atom_3_comp_id), fn(a.atom_id_3), float(a.value_angle), 1.0 / (float(a.value_angle_esd) ** 2)))
        _LINK_CACHE[key] = (bonds, angles)
    return _LINK_CACHE[key]


def _dihedral_deg(p0, p1, p2, p3):
    b0 = np.array(p0, dtype=np.float64) - np.array(p1, dtype=np.float64)
    b1 = np.array(p2, dtype=np.float64) - np.array(p1, dtype=np.float64)
    b2 = np.array(p3, dtype=np.float64) - np.array(p2, dtype=np.float64)
    b1_norm = float(np.linalg.norm(b1))
    if b1_norm == 0.0:
        return 180.0
    b1 /= b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def _extract_v34_fast(hierarchy):
    """
    Fast in-memory extraction of v34 bond and angle restraints (monomer library
    templates, peptide/RNA/DNA polymer links, disulfide bonds, and GPU CDL updates)
    without running full mmtbx.model.manager / pdb_interpretation.
    Returns None if the block contains altlocs, hydrogens, unknown monomers,
    ambiguous modified nucleotides, or non-polymer covalent links, falling back
    to _model(hierarchy, 'v34').
    """
    from iotbx.pdb.rna_dna_detection import residue_analysis
    import mmtbx.geometry_restraints.torsion_restraints.utils as torsion_utils
    from mmtbx.monomer_library.rna_sugar_pucker_analysis import evaluate as rna_evaluate
    from scipy.spatial import cKDTree

    h = hierarchy.deep_copy()
    h.merge_atoms_at_end_to_residues()
    h.sort_atoms_in_place()
    h.atoms().reset_i_seq()
    atoms = h.atoms()
    n_atoms = atoms.size()
    if n_atoms == 0:
        return None
    for a in atoms:
        e = a.element.strip().upper()
        if e:
            if e in ('H', 'D'):
                return None
        else:
            nm = a.name.strip().lstrip('0123456789')
            if nm[:1] in ('H', 'D') and a.parent().resname.strip() != nm:
                return None

    srv, rna_params, interp_params = _get_mon_lib()
    peptide_break_sq = 3.0 ** 2
    na_break_sq = 4.0 ** 2

    bonds_by_i0 = {}
    angle_iseqs_list = []
    angle_ideal_list = []
    angle_weight_list = []
    angle_origin_list = []
    cys_sg_atoms = []
    nonpoly_cov_xyz = []
    nonpoly_cov_res = []
    all_res_idx = np.full(n_atoms, -1, dtype=np.int32)
    cur_res_id = 0

    def add_bond(i0, i1, ideal, weight, origin=0):
        if i0 == i1:
            return
        if i0 > i1:
            i0, i1 = i1, i0
        lst = bonds_by_i0.get(i0)
        if lst is None:
            bonds_by_i0[i0] = [(i1, ideal, weight, origin)]
        else:
            for idx, (prev_i1, _, _, _) in enumerate(lst):
                if prev_i1 == i1:
                    return
            lst.append((i1, ideal, weight, origin))

    def add_angle(i0, i1, i2, ideal, weight, origin=0):
        if i0 == i1 or i1 == i2 or i0 == i2:
            return
        if i0 > i2:
            i0, i2 = i2, i0
        angle_iseqs_list.append((i0, i1, i2))
        angle_ideal_list.append(ideal)
        angle_weight_list.append(weight)
        angle_origin_list.append(origin)

    for chain in h.models()[0].chains():
        rgs = list(chain.residue_groups())
        n_rgs = len(rgs)
        rg_infos = []
        for rg in rgs:
            ags = rg.atom_groups()
            if len(ags) != 1 or ags[0].altloc.strip() != '':
                return None
            ag = ags[0]
            resname = ag.resname.strip()
            t_entry = _get_comp_template(resname)
            if t_entry is None:
                return None
            comp, _, _, _ = t_entry
            is_pep = comp.is_peptide()
            if not is_pep:
                ra1 = residue_analysis(residue_atoms=ag.atoms(), distance_tolerance=rna_params.bond_detection_distance_tolerance)
                is_na = (ra1.problems is None) or comp.is_rna_dna()
                if is_na and (ra1.problems is not None or comp.classification not in ('RNA', 'DNA')):
                    return None
            else:
                ra1 = None
                is_na = False
            if is_na:
                amap = {_norm_na_atom(a.name): a for a in ag.atoms()}
            else:
                amap = {a.name.strip(): a for a in ag.atoms()}
            for a in ag.atoms():
                all_res_idx[a.i_seq] = cur_res_id
            if not is_pep and not is_na and resname not in _NON_COVALENT_SOLVENT_IONS:
                for a in ag.atoms():
                    el = a.element.strip().upper()
                    if el in ('C', 'N', 'O', 'P', 'S', 'B', 'SE'):
                        nonpoly_cov_xyz.append(a.xyz)
                        nonpoly_cov_res.append(cur_res_id)
            cur_res_id += 1
            rg_infos.append((rg, ag, resname, comp, is_pep, is_na, ra1, amap))

        na_is_2p = [False] * n_rgs
        na_pucker_mod = [None] * n_rgs
        for idx_rg, (rg, ag, resname, comp, is_pep, is_na, ra1, amap) in enumerate(rg_infos):
            if is_na and ra1 is not None and ra1.is_rna:
                next_pdb_res = rg_infos[idx_rg + 1][1] if idx_rg + 1 < n_rgs else None
                ana = rna_evaluate(
                    params=rna_params,
                    residue_1_deoxy_ribo_atom_dict=ra1.deoxy_ribo_atom_dict,
                    residue_1_c1p_outbound_atom=ra1.c1p_outbound_atom,
                    residue_2_p_atom=next_pdb_res.get_atom('P') if next_pdb_res is not None else None,
                )
                is_2p = bool(ana is not None and ana.is_2p)
                na_is_2p[idx_rg] = is_2p
                mod_resname = torsion_utils.modernize_rna_resname(resname=ag.resname)
                if is_2p:
                    if mod_resname in ('  A', '  G'):
                        na_pucker_mod[idx_rg] = 'rna2p_pur'
                    elif mod_resname in ('  C', '  U'):
                        na_pucker_mod[idx_rg] = 'rna2p_pyr'
                    else:
                        na_pucker_mod[idx_rg] = 'rna2p'
                else:
                    if mod_resname in ('  A', '  G'):
                        na_pucker_mod[idx_rg] = 'rna3p_pur'
                    elif mod_resname in ('  C', '  U'):
                        na_pucker_mod[idx_rg] = 'rna3p_pyr'
                    else:
                        na_pucker_mod[idx_rg] = 'rna3p'

        prev_info = None
        for idx_rg, (rg, ag, resname, comp, is_pep, is_na, ra1, amap) in enumerate(rg_infos):
            if is_na:
                if 'O3T' in amap:
                    term_mod = 'p5*END'
                elif 'P' not in amap and 'OP1' not in amap and 'OP2' not in amap:
                    term_mod = '5*END'
                else:
                    term_mod = None
                t_entry = _get_na_template(resname, na_pucker_mod[idx_rg], term_mod)
                if t_entry is None:
                    return None
                _, b_list, a_list, valid_atoms = t_entry
            else:
                has_oxt = is_pep and ('OXT' in amap)
                t_entry = _get_comp_template(resname, has_oxt=has_oxt)
                if t_entry is None:
                    return None
                _, b_list, a_list, valid_atoms = t_entry
            if not (amap.keys() <= valid_atoms):
                return None

            for n1, n2, ideal, w in b_list:
                a1, a2 = amap.get(n1), amap.get(n2)
                if a1 is not None and a2 is not None:
                    add_bond(a1.i_seq, a2.i_seq, ideal, w, 0)

            if resname == 'CYS' and 'SG' in amap:
                cys_sg_atoms.append((amap['SG'].i_seq, amap['CA'].i_seq if 'CA' in amap else None, amap['CB'].i_seq if 'CB' in amap else None))

            # Inter-residue link angles come BEFORE the second residue's intra-residue monomer angles
            if rg.link_to_previous and prev_info is not None:
                prev_ag, prev_resname, prev_comp, prev_is_pep, prev_is_na, prev_amap, prev_2p = prev_info
                if prev_is_pep and is_pep:
                    c_atom = prev_amap.get('C')
                    n_atom = amap.get('N')
                    if c_atom is not None and n_atom is not None:
                        dx = c_atom.xyz[0] - n_atom.xyz[0]
                        dy = c_atom.xyz[1] - n_atom.xyz[1]
                        dz = c_atom.xyz[2] - n_atom.xyz[2]
                        if dx * dx + dy * dy + dz * dz <= 9.0:
                            if 'CN' in amap and _get_link_template('NMTRANS', norm_na=False) is not None:
                                link_id = 'NMTRANS'
                            elif resname == 'PRO':
                                link_id = 'PTRANS'
                            else:
                                link_id = 'TRANS'
                            l_entry = _get_link_template(link_id, norm_na=False)
                            if l_entry is not None:
                                lbonds, langles = l_entry
                                maps = (prev_amap, amap)
                                for c1, n1, c2, n2, ideal, w in lbonds:
                                    at1 = maps[c1 - 1].get(n1)
                                    at2 = maps[c2 - 1].get(n2)
                                    if at1 is not None and at2 is not None:
                                        add_bond(at1.i_seq, at2.i_seq, ideal, w, 0)
                                for c1, n1, c2, n2, c3, n3, ideal, w in langles:
                                    at1 = maps[c1 - 1].get(n1)
                                    at2 = maps[c2 - 1].get(n2)
                                    at3 = maps[c3 - 1].get(n3)
                                    if at1 is not None and at2 is not None and at3 is not None:
                                        add_angle(at1.i_seq, at2.i_seq, at3.i_seq, ideal, w, 0)
                elif prev_is_na and is_na:
                    o3_atom = prev_amap.get("O3'")
                    p_atom = amap.get('P')
                    if o3_atom is not None and p_atom is not None:
                        dx = o3_atom.xyz[0] - p_atom.xyz[0]
                        dy = o3_atom.xyz[1] - p_atom.xyz[1]
                        dz = o3_atom.xyz[2] - p_atom.xyz[2]
                        if dx * dx + dy * dy + dz * dz <= 9.0:
                            link_id = 'rna2p' if prev_2p else 'rna3p'
                            l_entry = _get_link_template(link_id, norm_na=True)
                            if l_entry is not None:
                                lbonds, langles = l_entry
                                maps = (prev_amap, amap)
                                for c1, n1, c2, n2, ideal, w in lbonds:
                                    at1 = maps[c1 - 1].get(n1)
                                    at2 = maps[c2 - 1].get(n2)
                                    if at1 is not None and at2 is not None:
                                        add_bond(at1.i_seq, at2.i_seq, ideal, w, 0)
                                for c1, n1, c2, n2, c3, n3, ideal, w in langles:
                                    at1 = maps[c1 - 1].get(n1)
                                    at2 = maps[c2 - 1].get(n2)
                                    at3 = maps[c3 - 1].get(n3)
                                    if at1 is not None and at2 is not None and at3 is not None:
                                        add_angle(at1.i_seq, at2.i_seq, at3.i_seq, ideal, w, 0)

            for n1, n2, n3, ideal, w in a_list:
                a1, a2, a3 = amap.get(n1), amap.get(n2), amap.get(n3)
                if a1 is not None and a2 is not None and a3 is not None:
                    add_angle(a1.i_seq, a2.i_seq, a3.i_seq, ideal, w, 0)

            prev_info = (ag, resname, comp, is_pep, is_na, amap, na_is_2p[idx_rg])

    sites = atoms.extract_xyz()
    xyz = np.asarray(sites.as_numpy_array(), dtype=np.float64)
    is_h = np.zeros(n_atoms, dtype=bool)

    if nonpoly_cov_xyz:
        tree_all = cKDTree(xyz)
        np_xyz = np.asarray(nonpoly_cov_xyz, dtype=np.float64)
        hits = tree_all.query_ball_point(np_xyz, 1.95)
        for q_idx, nbrs in enumerate(hits):
            r_q = nonpoly_cov_res[q_idx]
            for j in nbrs:
                if all_res_idx[j] != r_q:
                    return None

    if len(cys_sg_atoms) >= 2:
        sg_xyz = xyz[[x[0] for x in cys_sg_atoms]]
        tree = cKDTree(sg_xyz)
        pairs = sorted(tree.query_pairs(r=3.0))
        if pairs:
            used_sg = set()
            pairs_with_d = []
            for u, v in pairs:
                sg1 = cys_sg_atoms[u][0]
                sg2 = cys_sg_atoms[v][0]
                d2 = float(np.sum((xyz[sg1] - xyz[sg2]) ** 2))
                pairs_with_d.append((d2, u, v))
            pairs_with_d.sort()
            ss_w = 1.0 / (0.02 ** 2)
            ss_ang_w = 1.0 / (2.1 ** 2)
            for _, u, v in pairs_with_d:
                sg1, ca1, cb1 = cys_sg_atoms[u]
                sg2, ca2, cb2 = cys_sg_atoms[v]
                if sg1 in used_sg or sg2 in used_sg:
                    continue
                used_sg.add(sg1)
                used_sg.add(sg2)
                i_s, j_s = (sg1, sg2) if sg1 < sg2 else (sg2, sg1)
                ca_i, cb_i = (ca1, cb1) if sg1 < sg2 else (ca2, cb2)
                ca_j, cb_j = (ca2, cb2) if sg1 < sg2 else (ca1, cb1)
                add_bond(i_s, j_s, 2.031, ss_w, 1)
                if ca_i is not None and cb_i is not None and ca_j is not None and cb_j is not None:
                    add_angle(cb_i, i_s, j_s, 104.2, ss_ang_w, 1)
                    add_angle(i_s, j_s, cb_j, 104.2, ss_ang_w, 1)

    flat_bonds_iseqs = []
    flat_bonds_ideal = []
    flat_bonds_weight = []
    flat_bonds_origin = []
    for i0 in range(n_atoms):
        lst = bonds_by_i0.get(i0)
        if lst:
            for i1, ideal, w, orig in sorted(lst):
                flat_bonds_iseqs.append((i0, i1))
                flat_bonds_ideal.append(ideal)
                flat_bonds_weight.append(w)
                flat_bonds_origin.append(orig)

    if flat_bonds_iseqs:
        bond_iseqs = np.array(flat_bonds_iseqs, dtype=np.int32)
        bond_ideal = np.array(flat_bonds_ideal, dtype=np.float64)
        bond_weight = np.array(flat_bonds_weight, dtype=np.float64)
        bond_origin_id = np.array(flat_bonds_origin, dtype=np.uint8)
    else:
        bond_iseqs = np.empty((0, 2), dtype=np.int32)
        bond_ideal = np.empty(0, dtype=np.float64)
        bond_weight = np.empty(0, dtype=np.float64)
        bond_origin_id = np.empty(0, dtype=np.uint8)

    if angle_iseqs_list:
        angle_iseqs = np.array(angle_iseqs_list, dtype=np.int32)
        angle_ideal = np.array(angle_ideal_list, dtype=np.float64)
        angle_weight = np.array(angle_weight_list, dtype=np.float64)
        angle_origin_id = np.array(angle_origin_list, dtype=np.uint8)
    else:
        angle_iseqs = np.empty((0, 3), dtype=np.int32)
        angle_ideal = np.empty(0, dtype=np.float64)
        angle_weight = np.empty(0, dtype=np.float64)
        angle_origin_id = np.empty(0, dtype=np.uint8)

    bond_ideal, bond_weight, angle_ideal, angle_weight = apply_cdl_updates(
        h,
        bond_iseqs,
        bond_ideal,
        bond_weight,
        angle_iseqs,
        angle_ideal,
        angle_weight,
        angle_origin_id=angle_origin_id,
    )

    atom_tuples = [atom_tuple(a.fetch_labels()) for a in atoms]
    return {
        'xyz': xyz,
        'is_h': is_h,
        'atom_tuples': atom_tuples,
        'bond_iseqs': bond_iseqs,
        'bond_ideal': bond_ideal,
        'bond_weight': bond_weight,
        'bond_origin_id': bond_origin_id,
        'angle_iseqs': angle_iseqs,
        'angle_ideal': angle_ideal,
        'angle_weight': angle_weight,
        'angle_origin_id': angle_origin_id,
    }


def extract_restraints(hierarchy, definition):
    """
    Extract flat NumPy restraint proxy arrays from an iotbx hierarchy via mmtbx.model
    (or via the fast monomer-library + GPU CDL path for v34 when applicable).
    Used when running with cctbx in-process or when building cached restraint archives.
    """
    if definition == 'v34':
        fast = _extract_v34_fast(hierarchy)
        if fast is not None:
            return fast
    from cctbx import geometry_restraints
    from .geometry import _model

    model = _model(hierarchy, definition)
    h = model.get_hierarchy()
    atoms = h.atoms()
    if definition == 'v33':
        hd = model.get_hd_selection()
        is_h = np.asarray(hd, dtype=bool)
    else:
        is_h = np.array([e.strip().upper() in ('H', 'D') for e in atoms.extract_element()], dtype=bool)

    sites = atoms.extract_xyz()
    xyz = np.asarray(sites.as_numpy_array(), dtype=np.float64)
    grm = model.get_restraints_manager().geometry
    pair = grm.pair_proxies(sites_cart=sites, flags=geometry_restraints.flags.flags(bond=True)).bond_proxies
    if pair.asu.size():
        raise InputError('symmetry-related bonds in a block')
    bonds = pair.simple
    angles = grm.angle_proxies

    n_b = bonds.size()
    bond_iseqs = np.empty((n_b, 2), dtype=np.int32)
    bond_ideal = np.empty(n_b, dtype=np.float64)
    bond_weight = np.empty(n_b, dtype=np.float64)
    bond_origin_id = np.empty(n_b, dtype=np.uint8)
    for j, p in enumerate(bonds):
        bond_iseqs[j] = p.i_seqs
        bond_ideal[j] = p.distance_ideal
        bond_weight[j] = p.weight
        bond_origin_id[j] = p.origin_id

    n_a = angles.size()
    angle_iseqs = np.empty((n_a, 3), dtype=np.int32)
    angle_ideal = np.empty(n_a, dtype=np.float64)
    angle_weight = np.empty(n_a, dtype=np.float64)
    angle_origin_id = np.empty(n_a, dtype=np.uint8)
    for j, p in enumerate(angles):
        angle_iseqs[j] = p.i_seqs
        angle_ideal[j] = p.angle_ideal
        angle_weight[j] = p.weight
        angle_origin_id[j] = p.origin_id

    atom_tuples = [atom_tuple(a.fetch_labels()) for a in atoms]
    out = {
        'xyz': xyz,
        'is_h': is_h,
        'atom_tuples': atom_tuples,
        'bond_iseqs': bond_iseqs,
        'bond_ideal': bond_ideal,
        'bond_weight': bond_weight,
        'bond_origin_id': bond_origin_id,
        'angle_iseqs': angle_iseqs,
        'angle_ideal': angle_ideal,
        'angle_weight': angle_weight,
        'angle_origin_id': angle_origin_id,
    }
    if definition == 'v33':
        dp = grm.dihedral_proxies
        nd = dp.size()
        d_iseqs = np.empty((nd, 4), dtype=np.int32)
        d_ideal = np.empty(nd, dtype=np.float64)
        d_weight = np.empty(nd, dtype=np.float64)
        d_period = np.empty(nd, dtype=np.int32)
        d_slack = np.empty(nd, dtype=np.float64)
        d_limit = np.empty(nd, dtype=np.float64)
        d_top_out = np.empty(nd, dtype=bool)
        d_origin = np.empty(nd, dtype=np.uint8)
        d_alt = np.full((nd, 6), np.nan, dtype=np.float64)
        for j, p in enumerate(dp):
            d_iseqs[j] = p.i_seqs
            d_ideal[j] = p.angle_ideal
            d_weight[j] = p.weight
            d_period[j] = p.periodicity
            d_slack[j] = p.slack
            d_limit[j] = p.limit
            d_top_out[j] = p.top_out
            d_origin[j] = p.origin_id
            if p.alt_angle_ideals:
                for k, a in enumerate(p.alt_angle_ideals):
                    d_alt[j, k] = a

        cp = grm.chirality_proxies
        nc = cp.size()
        c_iseqs = np.empty((nc, 4), dtype=np.int32)
        c_ideal = np.empty(nc, dtype=np.float64)
        c_weight = np.empty(nc, dtype=np.float64)
        c_both = np.empty(nc, dtype=bool)
        c_origin = np.empty(nc, dtype=np.uint8)
        for j, p in enumerate(cp):
            c_iseqs[j] = p.i_seqs
            c_ideal[j] = p.volume_ideal
            c_weight[j] = p.weight
            c_both[j] = p.both_signs
            c_origin[j] = p.origin_id

        pp = grm.planarity_proxies
        npl = pp.size()
        max_pl = max((len(p.i_seqs) for p in pp), default=4)
        p_iseqs = np.full((npl, max_pl), -1, dtype=np.int32)
        p_weights = np.zeros((npl, max_pl), dtype=np.float64)
        p_origin = np.empty(npl, dtype=np.uint8)
        for j, p in enumerate(pp):
            k = len(p.i_seqs)
            p_iseqs[j, :k] = list(p.i_seqs)
            p_weights[j, :k] = list(p.weights)
            p_origin[j] = p.origin_id

        out.update({
            'dihedral_iseqs': d_iseqs,
            'dihedral_ideal': d_ideal,
            'dihedral_weight': d_weight,
            'dihedral_periodicity': d_period,
            'dihedral_slack': d_slack,
            'dihedral_limit': d_limit,
            'dihedral_top_out': d_top_out,
            'dihedral_origin_id': d_origin,
            'dihedral_alt_ideals': d_alt,
            'chirality_iseqs': c_iseqs,
            'chirality_ideal': c_ideal,
            'chirality_weight': c_weight,
            'chirality_both_signs': c_both,
            'chirality_origin_id': c_origin,
            'planarity_iseqs': p_iseqs,
            'planarity_weights': p_weights,
            'planarity_origin_id': p_origin,
        })
    return out


def score_from_restraints(definition, data, owned):
    """
    Score v34 or v33 bond and angle restraints on GPU from extracted restraint arrays.
    """
    if definition not in DEFINITIONS:
        raise ValueError('unknown geometry definition %r (known: %s)' % (definition, ', '.join(DEFINITIONS)))

    xyz = data['xyz']
    is_h = data['is_h']
    raw_tuples = data['atom_tuples']
    if isinstance(raw_tuples, np.ndarray):
        atom_tuples = [tuple(str(x) for x in row) for row in raw_tuples]
    else:
        atom_tuples = list(raw_tuples)

    bond_iseqs = data['bond_iseqs']
    bond_ideal = data['bond_ideal']
    bond_weight = data['bond_weight']
    bond_origin_id = data['bond_origin_id']

    angle_iseqs = data['angle_iseqs']
    angle_ideal = data['angle_ideal']
    angle_weight = data['angle_weight']
    angle_origin_id = data['angle_origin_id']

    # Check for degenerate angles (same atom twice)
    if len(angle_iseqs):
        bad = np.flatnonzero(
            (angle_iseqs[:, 0] == angle_iseqs[:, 1])
            | (angle_iseqs[:, 0] == angle_iseqs[:, 2])
            | (angle_iseqs[:, 1] == angle_iseqs[:, 2])
        )
        if bad.size:
            raise RuntimeError(
                'degenerate angle on atoms %s'
                % ', '.join(str(atom_tuples[int(i)]) for i in angle_iseqs[bad[0]])
            )

    # Precompute ownership and residue index per atom
    mine_cache = {}
    atom_owned = np.empty(len(atom_tuples), dtype=bool)
    res_key_to_idx = {}
    idx_to_res_key = []
    atom_res_idx = np.empty(len(atom_tuples), dtype=np.int32)

    for i, at in enumerate(atom_tuples):
        rk = (at[0], at[1])
        m = mine_cache.get(rk)
        if m is None:
            m = mine_cache[rk] = bool(owned(rk[0], rk[1]))
        atom_owned[i] = m
        r_idx = res_key_to_idx.get(rk)
        if r_idx is None:
            r_idx = res_key_to_idx[rk] = len(idx_to_res_key)
            idx_to_res_key.append(rk)
        atom_res_idx[i] = r_idx

    xyz_p = _bucket_xyz_np(xyz)
    # Evaluate bonds and angles on GPU via JAX
    b_model, b_delta, b_sigma, b_z = eval_bonds_jax(xyz_p, bond_iseqs, bond_ideal, bond_weight)
    a_model, a_delta, a_sigma, a_z = eval_angles_jax(xyz_p, angle_iseqs, angle_ideal, angle_weight)

    b_noh = ~is_h[bond_iseqs].any(axis=1) if len(bond_iseqs) else np.empty(0, dtype=bool)
    a_noh = ~is_h[angle_iseqs].any(axis=1) if len(angle_iseqs) else np.empty(0, dtype=bool)
    b_owned = atom_owned[bond_iseqs[:, 0]] if len(bond_iseqs) else np.empty(0, dtype=bool)
    a_owned = atom_owned[angle_iseqs[:, 0]] if len(angle_iseqs) else np.empty(0, dtype=bool)

    if definition == 'v34':
        b_keep = b_noh & np.isin(bond_origin_id, V34_SCOREABLE_ORIGIN_IDS) & b_owned
        a_keep = a_noh & np.isin(angle_origin_id, V34_SCOREABLE_ORIGIN_IDS) & a_owned

        n_res = max(len(idx_to_res_key), 1)
        b_res_bucket = _next_bucket(n_res)
        if len(bond_iseqs):
            nb = len(bond_iseqs)
            bb = _next_bucket(nb)
            bz2_p = np.pad(b_z * b_z, (0, bb - nb), mode="constant")
            bkeep_p = np.pad(b_keep, (0, bb - nb), mode="constant")
            bres_p = np.pad(atom_res_idx[bond_iseqs[:, 0]], (0, bb - nb), mode="constant")
            b_counts, b_z2 = call_gpu("accum_rmsz", bz2_p, bkeep_p, bres_p, b_res_bucket)
            b_counts, b_z2 = b_counts[:n_res], b_z2[:n_res]
        else:
            b_counts = np.zeros(n_res, dtype=np.int32)
            b_z2 = np.zeros(n_res, dtype=np.float64)

        if len(angle_iseqs):
            na = len(angle_iseqs)
            ba = _next_bucket(na)
            az2_p = np.pad(a_z * a_z, (0, ba - na), mode="constant")
            akeep_p = np.pad(a_keep, (0, ba - na), mode="constant")
            ares_p = np.pad(atom_res_idx[angle_iseqs[:, 0]], (0, ba - na), mode="constant")
            a_counts, a_z2 = call_gpu("accum_rmsz", az2_p, akeep_p, ares_p, b_res_bucket)
            a_counts, a_z2 = a_counts[:n_res], a_z2[:n_res]
        else:
            a_counts = np.zeros(n_res, dtype=np.int32)
            a_z2 = np.zeros(n_res, dtype=np.float64)

        residues_out = {}
        active_res = np.flatnonzero((b_counts > 0) | (a_counts > 0))
        for r_idx in active_res:
            rk = idx_to_res_key[int(r_idx)]
            residues_out[rk] = [
                int(b_counts[r_idx]),
                float(b_z2[r_idx]),
                int(a_counts[r_idx]),
                float(a_z2[r_idx]),
            ]

        b_outliers = []
        for j in np.flatnonzero(b_keep & (np.abs(b_z) > V34_CUTOFF)):
            i0, i1 = int(bond_iseqs[j, 0]), int(bond_iseqs[j, 1])
            b_outliers.append(
                (
                    atom_tuples[i0],
                    atom_tuples[i1],
                    float(bond_ideal[j] - b_delta[j]),
                    float(bond_ideal[j]),
                    float(b_sigma[j]),
                    float(b_z[j]),
                )
            )

        a_outliers = []
        for j in np.flatnonzero(a_keep & (np.abs(a_z) > V34_CUTOFF)):
            i0, i1, i2 = int(angle_iseqs[j, 0]), int(angle_iseqs[j, 1]), int(angle_iseqs[j, 2])
            a_outliers.append(
                (
                    atom_tuples[i0],
                    atom_tuples[i1],
                    atom_tuples[i2],
                    float(angle_ideal[j] - a_delta[j]),
                    float(angle_ideal[j]),
                    float(a_sigma[j]),
                    float(a_z[j]),
                )
            )

        return {
            'residues': residues_out,
            'bonds': {'total': int(b_keep.sum()), 'outliers': b_outliers},
            'angles': {'total': int(a_keep.sum()), 'outliers': a_outliers},
        }

    # definition == 'v33'
    b_total_mask = b_noh & np.isin(bond_origin_id, V33_COVALENT_ORIGIN_IDS) & b_owned
    a_total_mask = a_noh & np.isin(angle_origin_id, V33_COVALENT_ORIGIN_IDS) & a_owned

    b_score = np.abs(b_z)
    a_score = np.abs(a_z)
    b_out_mask = b_noh & (bond_origin_id == V33_OUTLIER_ORIGIN_ID) & b_owned & (b_score > V33_CUTOFF)
    a_out_mask = a_noh & (angle_origin_id == V33_OUTLIER_ORIGIN_ID) & a_owned & (a_score > V33_CUTOFF)

    # Sort descending by score (matching restraints.combined with reverse_sort=True)
    b_out_idx = np.flatnonzero(b_out_mask)
    if b_out_idx.size:
        b_out_idx = b_out_idx[np.argsort(-b_score[b_out_idx], kind='stable')]
    b_outliers = [
        (
            atom_tuples[int(bond_iseqs[j, 0])],
            atom_tuples[int(bond_iseqs[j, 1])],
            float(b_model[j]),
            float(bond_ideal[j]),
            float(b_score[j]),
        )
        for j in b_out_idx
    ]

    a_out_idx = np.flatnonzero(a_out_mask)
    if a_out_idx.size:
        a_out_idx = a_out_idx[np.argsort(-a_score[a_out_idx], kind='stable')]
    a_outliers = [
        (
            atom_tuples[int(angle_iseqs[j, 0])],
            atom_tuples[int(angle_iseqs[j, 1])],
            atom_tuples[int(angle_iseqs[j, 2])],
            float(a_model[j]),
            float(angle_ideal[j]),
            float(a_score[j]),
        )
        for j in a_out_idx
    ]

    # Dihedrals (v33)
    d_iseqs = data.get('dihedral_iseqs', np.empty((0, 4), dtype=np.int32))
    if len(d_iseqs):
        d_ideal = data['dihedral_ideal']
        d_weight = data['dihedral_weight']
        d_period = data['dihedral_periodicity']
        d_slack = data['dihedral_slack']
        d_limit = data['dihedral_limit']
        d_top_out = data['dihedral_top_out']
        d_origin = data['dihedral_origin_id']
        d_alt = data['dihedral_alt_ideals']

        d_noh = ~is_h[d_iseqs].any(axis=1)
        d_owned = atom_owned[d_iseqs[:, 0]]
        d_total_mask = d_noh & d_owned

        d_model, _, _, d_score = eval_dihedrals_jax(
            xyz_p, d_iseqs, d_ideal, d_weight, d_period, d_slack, d_limit, d_top_out, d_alt
        )
        d_out_mask = d_total_mask & (d_origin == V33_OUTLIER_ORIGIN_ID) & (d_score > V33_CUTOFF)
        d_out_idx = np.flatnonzero(d_out_mask)
        if d_out_idx.size:
            d_out_idx = d_out_idx[np.argsort(-d_score[d_out_idx], kind='stable')]
        d_outliers = [
            (
                atom_tuples[int(d_iseqs[j, 0])],
                atom_tuples[int(d_iseqs[j, 1])],
                atom_tuples[int(d_iseqs[j, 2])],
                atom_tuples[int(d_iseqs[j, 3])],
                float(d_model[j]),
                float(d_ideal[j]),
                float(d_score[j]),
            )
            for j in d_out_idx
        ]
        d_total = int(d_total_mask.sum())
    else:
        d_total, d_outliers = 0, []

    # Chiralities (v33)
    c_iseqs = data.get('chirality_iseqs', np.empty((0, 4), dtype=np.int32))
    if len(c_iseqs):
        c_ideal = data['chirality_ideal']
        c_weight = data['chirality_weight']
        c_both = data['chirality_both_signs']
        c_origin = data['chirality_origin_id']

        c_noh = ~is_h[c_iseqs].any(axis=1)
        c_owned = atom_owned[c_iseqs[:, 0]]
        c_total_mask = c_noh & c_owned

        c_model, _, _, c_score = eval_chiralities_jax(xyz_p, c_iseqs, c_ideal, c_weight, c_both)
        c_out_mask = c_total_mask & (c_origin == V33_OUTLIER_ORIGIN_ID) & (c_score > V33_CUTOFF)
        c_out_idx = np.flatnonzero(c_out_mask)
        if c_out_idx.size:
            c_out_idx = c_out_idx[np.argsort(-c_score[c_out_idx], kind='stable')]
        c_outliers = [
            (
                atom_tuples[int(c_iseqs[j, 0])],
                atom_tuples[int(c_iseqs[j, 1])],
                atom_tuples[int(c_iseqs[j, 2])],
                atom_tuples[int(c_iseqs[j, 3])],
                float(c_model[j]),
                float(c_ideal[j]),
                float(c_score[j]),
            )
            for j in c_out_idx
        ]
        c_total = int(c_total_mask.sum())
    else:
        c_total, c_outliers = 0, []

    # Planarities (v33)
    # Matching shared_planarity_proxy_select (~hd): strip H atoms from each plane
    # and keep planes with > 3 non-H atoms.
    p_iseqs_raw = data.get('planarity_iseqs', np.empty((0, 4), dtype=np.int32))
    if len(p_iseqs_raw):
        p_weights_raw = data['planarity_weights']
        if is_h.any():
            valid_noh = (p_iseqs_raw >= 0) & (~is_h[np.maximum(p_iseqs_raw, 0)])
            counts_noh = valid_noh.sum(axis=1)
            keep_planes = counts_noh > 3
            p_iseqs_sub = p_iseqs_raw[keep_planes]
            p_weights_sub = p_weights_raw[keep_planes]
            valid_sub = valid_noh[keep_planes]
            n_keep = len(p_iseqs_sub)
            m_cols = p_iseqs_raw.shape[1]
            p_iseqs = np.full((n_keep, m_cols), -1, dtype=np.int32)
            p_weights = np.zeros((n_keep, m_cols), dtype=np.float64)
            for j in range(n_keep):
                v_j = valid_sub[j]
                k_j = int(v_j.sum())
                p_iseqs[j, :k_j] = p_iseqs_sub[j, v_j]
                p_weights[j, :k_j] = p_weights_sub[j, v_j]
        else:
            valid_raw = (p_iseqs_raw >= 0).sum(axis=1) > 3
            p_iseqs = p_iseqs_raw[valid_raw]
            p_weights = p_weights_raw[valid_raw]

        if len(p_iseqs):
            p_owned = atom_owned[p_iseqs[:, 0]]
            p_total = int(p_owned.sum())
            p_rms, p_dmax, _, p_score = eval_planarities_jax(xyz_p, p_iseqs, p_weights)
            p_out_mask = p_owned & (p_score > V33_CUTOFF)
            p_out_idx = np.flatnonzero(p_out_mask)
            if p_out_idx.size:
                p_out_idx = p_out_idx[np.argsort(-p_score[p_out_idx], kind='stable')]
            p_outliers = []
            for j in p_out_idx:
                row_iseqs = p_iseqs[j]
                atoms_j = tuple(atom_tuples[int(i)] for i in row_iseqs if i >= 0)
                p_outliers.append(
                    atoms_j + (float(p_rms[j]), float(p_dmax[j]), float(p_score[j]))
                )
        else:
            p_total, p_outliers = 0, []
    else:
        p_total, p_outliers = 0, []

    return {
        'bonds': {'total': int(b_total_mask.sum()), 'outliers': b_outliers},
        'angles': {'total': int(a_total_mask.sum()), 'outliers': a_outliers},
        'dihedrals': {'total': d_total, 'outliers': d_outliers},
        'chiralities': {'total': c_total, 'outliers': c_outliers},
        'planarities': {'total': p_total, 'outliers': p_outliers},
    }


def score(definition, hierarchy, owned, restraint_data=None):
    """
    The bonds and angles of hierarchy whose first atom is owned, by definition
    ('v34' or 'v33'), evaluated on GPU via JAX.
    If restraint_data is provided (or pre-extracted), skips mmtbx.model restraint construction.
    """
    if restraint_data is None:
        restraint_data = extract_restraints(hierarchy, definition)
    return score_from_restraints(definition, restraint_data, owned)
