"""
GPU-accelerated covalent bond and angle geometry validation (v34 CDL + v33)
in JAX (float64), matching remolprobity.geometry.score.
"""
import functools
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from .ids import atom_tuple, residue_key
from .io import InputError
from .jax_residues import _next_bucket, dihedral_angles_jax

jax.config.update("jax_enable_x64", True)

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

DEFAULT_CDL_NPZ_PATH = os.environ.get(
    "REMOLPROBITY_JAX_CDL",
    os.path.expanduser("~/.cache/remolprobity/jax_cdl.npz"),
)

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
def get_cdl_tables(npz_path=DEFAULT_CDL_NPZ_PATH):
    """Load the Conformation-Dependent Library (CDL) tables onto GPU."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"JAX CDL table archive not found at {npz_path}. "
            "Run remolprobity.jax_geometry.build_cdl_npz() in the cctbx environment."
        )
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
        "cdl_grid": jnp.asarray(data["cdl_grid"], dtype=jnp.float64),
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


@jax.jit
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


def eval_bonds_jax(xyz, iseqs, ideal, weight):
    """Bucket-padded wrapper around _eval_bonds_kernel."""
    n = len(iseqs)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    bucket = _next_bucket(n)
    xyz_j = jnp.asarray(xyz, dtype=jnp.float64)
    if bucket > n:
        pad = bucket - n
        iseqs_p = np.pad(iseqs, ((0, pad), (0, 0)), mode="constant")
        ideal_p = np.pad(ideal, (0, pad), mode="constant")
        weight_p = np.pad(weight, (0, pad), mode="constant", constant_values=1.0)
        m, d, s, z = _eval_bonds_kernel(
            xyz_j,
            jnp.asarray(iseqs_p, dtype=jnp.int32),
            jnp.asarray(ideal_p, dtype=jnp.float64),
            jnp.asarray(weight_p, dtype=jnp.float64),
        )
        return np.asarray(m[:n]), np.asarray(d[:n]), np.asarray(s[:n]), np.asarray(z[:n])
    m, d, s, z = _eval_bonds_kernel(
        xyz_j,
        jnp.asarray(iseqs, dtype=jnp.int32),
        jnp.asarray(ideal, dtype=jnp.float64),
        jnp.asarray(weight, dtype=jnp.float64),
    )
    return np.asarray(m), np.asarray(d), np.asarray(s), np.asarray(z)


@jax.jit
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


def eval_angles_jax(xyz, iseqs, ideal, weight):
    """Bucket-padded wrapper around _eval_angles_kernel."""
    n = len(iseqs)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    bucket = _next_bucket(n)
    xyz_j = jnp.asarray(xyz, dtype=jnp.float64)
    if bucket > n:
        pad = bucket - n
        iseqs_p = np.pad(iseqs, ((0, pad), (0, 0)), mode="constant")
        ideal_p = np.pad(ideal, (0, pad), mode="constant")
        weight_p = np.pad(weight, (0, pad), mode="constant", constant_values=1.0)
        m, d, s, z = _eval_angles_kernel(
            xyz_j,
            jnp.asarray(iseqs_p, dtype=jnp.int32),
            jnp.asarray(ideal_p, dtype=jnp.float64),
            jnp.asarray(weight_p, dtype=jnp.float64),
        )
        return np.asarray(m[:n]), np.asarray(d[:n]), np.asarray(s[:n]), np.asarray(z[:n])
    m, d, s, z = _eval_angles_kernel(
        xyz_j,
        jnp.asarray(iseqs, dtype=jnp.int32),
        jnp.asarray(ideal, dtype=jnp.float64),
        jnp.asarray(weight, dtype=jnp.float64),
    )
    return np.asarray(m), np.asarray(d), np.asarray(s), np.asarray(z)


@functools.partial(jax.jit, static_argnums=(3,))
def _accumulate_residue_rmsz_kernel(z_sq, keep_mask, res_idx, n_residues):
    """Accumulate per-residue count and sum(z^2) on GPU via segment_sum."""
    counts = jax.ops.segment_sum(
        keep_mask.astype(jnp.int32), res_idx, num_segments=n_residues
    )
    sums = jax.ops.segment_sum(
        jnp.where(keep_mask, z_sq, 0.0), res_idx, num_segments=n_residues
    )
    return counts, sums


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


@jax.jit
def _cdl_lookup_kernel(group_idx, phi_idx, psi_idx, cdl_grid):
    """Gather the 24 CDL target/sigma columns for each (group, phi_bin, psi_bin) on GPU."""
    return cdl_grid[group_idx, phi_idx, psi_idx]


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
    tables = get_cdl_tables()
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
    all_angles = np.asarray(
        dihedral_angles_jax(
            all_quads[:, 0], all_quads[:, 1], all_quads[:, 2], all_quads[:, 3]
        )
    )
    n_t = len(threes_list)
    phis = all_angles[:n_t]
    psis = all_angles[n_t : 2 * n_t]
    omegas = all_angles[2 * n_t :]

    # Batch lookup CDL grid on GPU
    group_indices = np.zeros(n_t, dtype=np.int32)
    phi_indices = np.zeros(n_t, dtype=np.int32)
    psi_indices = np.zeros(n_t, dtype=np.int32)
    valid_group = np.zeros(n_t, dtype=bool)

    for idx, (r0, r1, r2) in enumerate(threes_list):
        g = _get_res_type_group(r1[0], r2[0], tables)
        if g is not None:
            valid_group[idx] = True
            group_indices[idx] = tables["group_to_idx"][g]
            phi_indices[idx] = _round_to_ten_index(phis[idx])
            psi_indices[idx] = _round_to_ten_index(psis[idx])

    cdl_batch = np.asarray(
        _cdl_lookup_kernel(
            jnp.asarray(group_indices, dtype=jnp.int32),
            jnp.asarray(phi_indices, dtype=jnp.int32),
            jnp.asarray(psi_indices, dtype=jnp.int32),
            tables["cdl_grid"],
        )
    )

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


def extract_restraints(hierarchy, definition):
    """
    Extract flat NumPy restraint proxy arrays from an iotbx hierarchy via mmtbx.model.
    Used when running with cctbx in-process or when building cached restraint archives.
    """
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

    # Evaluate bonds and angles on GPU via JAX
    b_model, b_delta, b_sigma, b_z = eval_bonds_jax(xyz, bond_iseqs, bond_ideal, bond_weight)
    a_model, a_delta, a_sigma, a_z = eval_angles_jax(xyz, angle_iseqs, angle_ideal, angle_weight)

    b_noh = ~is_h[bond_iseqs].any(axis=1) if len(bond_iseqs) else np.empty(0, dtype=bool)
    a_noh = ~is_h[angle_iseqs].any(axis=1) if len(angle_iseqs) else np.empty(0, dtype=bool)
    b_owned = atom_owned[bond_iseqs[:, 0]] if len(bond_iseqs) else np.empty(0, dtype=bool)
    a_owned = atom_owned[angle_iseqs[:, 0]] if len(angle_iseqs) else np.empty(0, dtype=bool)

    if definition == 'v34':
        b_keep = b_noh & np.isin(bond_origin_id, V34_SCOREABLE_ORIGIN_IDS) & b_owned
        a_keep = a_noh & np.isin(angle_origin_id, V34_SCOREABLE_ORIGIN_IDS) & a_owned

        n_res = max(len(idx_to_res_key), 1)
        if len(bond_iseqs):
            b_counts, b_z2 = _accumulate_residue_rmsz_kernel(
                jnp.asarray(b_z * b_z, dtype=jnp.float64),
                jnp.asarray(b_keep, dtype=jnp.bool_),
                jnp.asarray(atom_res_idx[bond_iseqs[:, 0]], dtype=jnp.int32),
                n_res,
            )
            b_counts, b_z2 = np.asarray(b_counts), np.asarray(b_z2)
        else:
            b_counts = np.zeros(n_res, dtype=np.int32)
            b_z2 = np.zeros(n_res, dtype=np.float64)

        if len(angle_iseqs):
            a_counts, a_z2 = _accumulate_residue_rmsz_kernel(
                jnp.asarray(a_z * a_z, dtype=jnp.float64),
                jnp.asarray(a_keep, dtype=jnp.bool_),
                jnp.asarray(atom_res_idx[angle_iseqs[:, 0]], dtype=jnp.int32),
                n_res,
            )
            a_counts, a_z2 = np.asarray(a_counts), np.asarray(a_z2)
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

    return {
        'bonds': {'total': int(b_total_mask.sum()), 'outliers': b_outliers},
        'angles': {'total': int(a_total_mask.sum()), 'outliers': a_outliers},
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
