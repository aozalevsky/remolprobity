"""
Unit and oracle-parity tests for GPU covalent bond & angle geometry validation
(remolprobity.jax_geometry: v34 CDL + v33).
"""
import glob
import os
import pickle

import numpy as np
import pytest

from remolprobity import compare, io, jax_geometry, jax_io, result

CORPUS_DIR = os.environ.get(
    "REMOLPROBITY_CORPUS", os.path.expanduser("~/.cache/remolprobity/corpus")
)
REF_DIR = os.path.join(CORPUS_DIR, "reference")
CACHE_DIR = os.path.join(CORPUS_DIR, "restraints_cache")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_restraint_npz(npz_path, defn):
    d = np.load(npz_path, allow_pickle=False)
    out = {
        "xyz": d[f"{defn}_xyz"],
        "is_h": d[f"{defn}_is_h"],
        "atom_tuples": d[f"{defn}_atom_tuples"],
        "bond_iseqs": d[f"{defn}_bond_iseqs"],
        "bond_ideal": d[f"{defn}_bond_ideal"],
        "bond_weight": d[f"{defn}_bond_weight"],
        "bond_origin_id": d[f"{defn}_bond_origin_id"],
        "angle_iseqs": d[f"{defn}_angle_iseqs"],
        "angle_ideal": d[f"{defn}_angle_ideal"],
        "angle_weight": d[f"{defn}_angle_weight"],
        "angle_origin_id": d[f"{defn}_angle_origin_id"],
    }
    if defn == "v33" and "v33_dihedral_iseqs" in d:
        out.update({
            "dihedral_iseqs": d["v33_dihedral_iseqs"],
            "dihedral_ideal": d["v33_dihedral_ideal"],
            "dihedral_weight": d["v33_dihedral_weight"],
            "dihedral_periodicity": d["v33_dihedral_periodicity"],
            "dihedral_slack": d["v33_dihedral_slack"],
            "dihedral_limit": d["v33_dihedral_limit"],
            "dihedral_top_out": d["v33_dihedral_top_out"],
            "dihedral_origin_id": d["v33_dihedral_origin_id"],
            "dihedral_alt_ideals": d["v33_dihedral_alt_ideals"],
            "chirality_iseqs": d["v33_chirality_iseqs"],
            "chirality_ideal": d["v33_chirality_ideal"],
            "chirality_weight": d["v33_chirality_weight"],
            "chirality_both_signs": d["v33_chirality_both_signs"],
            "chirality_origin_id": d["v33_chirality_origin_id"],
            "planarity_iseqs": d["v33_planarity_iseqs"],
            "planarity_weights": d["v33_planarity_weights"],
            "planarity_origin_id": d["v33_planarity_origin_id"],
        })
    return out


def _find_cached_entries():
    if not os.path.isdir(REF_DIR) or not os.path.isdir(CACHE_DIR):
        return []
    by_name = {}
    for pkl_path in sorted(glob.glob(os.path.join(REF_DIR, "*.pkl"))):
        base = os.path.basename(pkl_path)
        name = base.split(".")[0]
        if name in by_name and not base.endswith(".3c1d5676313508f6.pkl"):
            continue
        npz_path = os.path.join(CACHE_DIR, f"{name}.npz")
        if not os.path.exists(npz_path):
            continue
        model_path = None
        for ext in (".cif", ".pdb"):
            cand = os.path.join(CORPUS_DIR, name + ext)
            if os.path.exists(cand):
                model_path = cand
                break
        if model_path is not None:
            by_name[name] = (name, model_path, pkl_path, npz_path)
    return [by_name[k] for k in sorted(by_name)]


CACHED_ENTRIES = _find_cached_entries()
CDL_ENTRIES = [e for e in CACHED_ENTRIES if e[0] != "9a1n_100k"]


@pytest.mark.parametrize(
    "name,model_path,pkl_path,npz_path",
    CDL_ENTRIES,
    ids=[e[0] for e in CDL_ENTRIES],
)
def test_cdl_updates_match_cctbx(name, model_path, pkl_path, npz_path):
    """
    Starting from base (cdl=False) restraint proxies, jax_geometry.apply_cdl_updates
    (using GPU dihedral evaluation + GPU CDL grid lookup) reproduces cctbx's v34
    CDL ideals and weights to < 1e-12.
    """
    d = np.load(npz_path, allow_pickle=False)
    if "base_bond_ideal" not in d:
        pytest.skip(f"No base restraints in {npz_path}")

    h = jax_io.read_hierarchy(model_path)
    h.merge_atoms_at_end_to_residues()
    h.reset_i_seq()
    b_ideal, b_weight, a_ideal, a_weight = jax_geometry.apply_cdl_updates(
        h,
        d["v34_bond_iseqs"],
        d["base_bond_ideal"],
        d["base_bond_weight"],
        d["v34_angle_iseqs"],
        d["base_angle_ideal"],
        d["base_angle_weight"],
        d["v34_angle_origin_id"],
        cis_pro_eh99=True,
    )
    is_h = d["v34_is_h"]
    b_noh = ~is_h[d["v34_bond_iseqs"]].any(axis=1)
    a_noh = ~is_h[d["v34_angle_iseqs"]].any(axis=1)
    assert np.max(np.abs(b_ideal[b_noh] - d["v34_bond_ideal"][b_noh])) < 1e-12
    assert np.max(np.abs(b_weight[b_noh] - d["v34_bond_weight"][b_noh])) < 1e-12
    assert np.max(np.abs(a_ideal[a_noh] - d["v34_angle_ideal"][a_noh])) < 1e-12
    assert np.max(np.abs(a_weight[a_noh] - d["v34_angle_weight"][a_noh])) < 1e-12


@pytest.mark.parametrize(
    "name,model_path,pkl_path,npz_path",
    CACHED_ENTRIES,
    ids=[e[0] for e in CACHED_ENTRIES],
)
def test_jax_geometry_matches_oracle_reference(name, model_path, pkl_path, npz_path):
    """
    Verify that jax_geometry.score('v34', ...) and jax_geometry.score('v33', ...)
    match the stock MolProbity oracle reference (totals, outliers, and per-residue
    v34 RMSZ sums) with 0 differences at tol=1e-6.
    """
    with open(pkl_path, "rb") as f:
        ref = pickle.load(f)

    v34_data = _load_restraint_npz(npz_path, "v34")
    v33_data = _load_restraint_npz(npz_path, "v33")

    g34 = jax_geometry.score("v34", None, lambda c, r: True, restraint_data=v34_data)
    g33 = jax_geometry.score("v33", None, lambda c, r: True, restraint_data=v33_data)

    m = io.read_model(model_path)
    assembled = result.assemble(
        m.residues,
        m.resnames,
        [],
        [],
        [],
        0,
        {"v34": g34, "v33": g33},
        g34["residues"],
        {},
        [],
        [],
        [],
    )

    assert compare.differences(ref["geometry"]["v34"], assembled["geometry"]["v34"], tol=1e-6) == []
    assert compare.differences(ref["geometry"]["v33"], assembled["geometry"]["v33"], tol=1e-6) == []
    assert compare.differences(ref["residues"], assembled["residues"], tol=1e-6) == []


def test_jax_geometry_splits_by_ownership():
    """Verify block ownership partitioning on 1crn.pdb."""
    npz_path = os.path.join(CACHE_DIR, "1crn.npz")
    if not os.path.exists(npz_path):
        pytest.skip("1crn.npz not cached")
    model_path = os.path.join(DATA_DIR, "1crn.pdb")
    m = io.read_model(model_path)
    half = set(m.residues[: len(m.residues) // 2])

    for defn in ("v34", "v33"):
        data = _load_restraint_npz(npz_path, defn)
        whole = jax_geometry.score(defn, None, lambda c, r: True, restraint_data=data)
        p1 = jax_geometry.score(defn, None, lambda c, r: (c, r) in half, restraint_data=data)
        p2 = jax_geometry.score(defn, None, lambda c, r: (c, r) not in half, restraint_data=data)

        for kind in result.KINDS[defn]:
            assert whole[kind]["total"] == p1[kind]["total"] + p2[kind]["total"]
            assert sorted(whole[kind]["outliers"]) == sorted(p1[kind]["outliers"] + p2[kind]["outliers"])
        if defn == "v34":
            assert compare.differences(whole["residues"], {**p1["residues"], **p2["residues"]}, tol=1e-12) == []


def test_jax_geometry_rejects_degenerate_angle():
    """Verify that an angle restraint with repeated atom indices raises RuntimeError."""
    data = {
        "xyz": np.zeros((3, 3), dtype=np.float64),
        "is_h": np.zeros(3, dtype=bool),
        "atom_tuples": [
            ("A", "   1 ", "ALA", " N  ", ""),
            ("A", "   1 ", "ALA", " CA ", ""),
            ("A", "   1 ", "ALA", " C  ", ""),
        ],
        "bond_iseqs": np.zeros((0, 2), dtype=np.int32),
        "bond_ideal": np.zeros(0, dtype=np.float64),
        "bond_weight": np.zeros(0, dtype=np.float64),
        "bond_origin_id": np.zeros(0, dtype=np.uint8),
        "angle_iseqs": np.array([[0, 1, 0]], dtype=np.int32),
        "angle_ideal": np.array([110.0], dtype=np.float64),
        "angle_weight": np.array([1.0], dtype=np.float64),
        "angle_origin_id": np.array([0], dtype=np.uint8),
    }
    with pytest.raises(RuntimeError, match="degenerate angle"):
        jax_geometry.score("v34", None, lambda c, r: True, restraint_data=data)
