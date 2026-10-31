"""
Unit and oracle-parity tests for GPU all-atom clash detection (probe)
(remolprobity.jax_clashes).
"""
import glob
import os
import pickle

import numpy as np
import pytest
from scipy.spatial import cKDTree

from remolprobity import compare, io, jax_clashes, result
from remolprobity.clashes import MarginError

CORPUS_DIR = os.environ.get(
    "REMOLPROBITY_CORPUS", os.path.expanduser("~/.cache/remolprobity/corpus")
)
REF_DIR = os.path.join(CORPUS_DIR, "reference")
WITHH_DIR = os.path.join(CORPUS_DIR, "withh_cache")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _find_cached_withh_entries():
    if not os.path.isdir(REF_DIR) or not os.path.isdir(WITHH_DIR):
        return []
    by_name = {}
    for pkl_path in sorted(glob.glob(os.path.join(REF_DIR, "*.pkl"))):
        base = os.path.basename(pkl_path)
        name = base.split(".")[0]
        if name in by_name and not base.endswith(".742d1a0065642254.pkl"):
            continue
        withh_pdb = os.path.join(WITHH_DIR, f"{name}.withh.pdb")
        meta_pkl = os.path.join(WITHH_DIR, f"{name}.meta.pkl")
        if os.path.exists(withh_pdb) and os.path.exists(meta_pkl):
            by_name[name] = (name, pkl_path, withh_pdb, meta_pkl)
    return [by_name[k] for k in sorted(by_name)]


CACHED_WITHH = _find_cached_withh_entries()


@pytest.mark.parametrize(
    "name,pkl_path,withh_pdb,meta_pkl",
    CACHED_WITHH,
    ids=[e[0] for e in CACHED_WITHH],
)
def test_jax_clashes_matches_oracle_reference(name, pkl_path, withh_pdb, meta_pkl):
    """
    Verify that jax_clashes.score_from_withh reproduces stock MolProbity's
    clashscore, probe_atoms, and clash records with 0 differences at tol=1e-6.
    """
    with open(pkl_path, "rb") as f:
        ref = pickle.load(f)
    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)

    res = jax_clashes.score_from_withh(
        withh_pdb,
        names=meta["names"],
        resnames=meta["resnames"],
        use_segids=meta["use_segids"],
        model_id=meta["model_id"],
    )
    n_atoms = res["probe_atoms"]
    records = sorted(res["records"], key=lambda c: (c[0], c[1], c[2], c[3]))
    clashscore = len(records) * 1000.0 / n_atoms if n_atoms else 0.0
    got = {"clashscore": clashscore, "atoms": n_atoms, "records": records}

    assert compare.differences(ref["clashes"], got, tol=1e-6) == []


def test_jax_clashes_splits_by_ownership():
    """Verify block ownership partitioning on 1crn."""
    withh_pdb = os.path.join(WITHH_DIR, "1crn.withh.pdb")
    meta_pkl = os.path.join(WITHH_DIR, "1crn.meta.pkl")
    if not os.path.exists(withh_pdb) or not os.path.exists(meta_pkl):
        pytest.skip("1crn withh_cache not found")

    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)
    m = io.read_model(os.path.join(DATA_DIR, "1crn.pdb"))
    half = set(m.residues[: len(m.residues) // 2])

    whole = jax_clashes.score_from_withh(
        withh_pdb,
        names=meta["names"],
        resnames=meta["resnames"],
        owned=lambda c, r: True,
        use_segids=meta["use_segids"],
        model_id=meta["model_id"],
    )
    p1 = jax_clashes.score_from_withh(
        withh_pdb,
        names=meta["names"],
        resnames=meta["resnames"],
        owned=lambda c, r: (c, r) in half,
        use_segids=meta["use_segids"],
        model_id=meta["model_id"],
    )
    p2 = jax_clashes.score_from_withh(
        withh_pdb,
        names=meta["names"],
        resnames=meta["resnames"],
        owned=lambda c, r: (c, r) not in half,
        use_segids=meta["use_segids"],
        model_id=meta["model_id"],
    )

    assert whole["probe_atoms"] == p1["probe_atoms"] + p2["probe_atoms"]
    assert sorted(whole["records"]) == sorted(p1["records"] + p2["records"])


def test_jax_clashes_margin_error():
    """Verify MarginError is raised when a clash reaches within REACH of the margin."""
    withh_pdb = os.path.join(WITHH_DIR, "1ehz.withh.pdb")
    meta_pkl = os.path.join(WITHH_DIR, "1ehz.meta.pkl")
    if not os.path.exists(withh_pdb) or not os.path.exists(meta_pkl):
        pytest.skip("1ehz withh_cache not found")

    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)

    far_tree = cKDTree(np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float64))
    with pytest.raises(MarginError):
        jax_clashes.score_from_withh(
            withh_pdb,
            names=meta["names"],
            resnames=meta["resnames"],
            use_segids=meta["use_segids"],
            model_id=meta["model_id"],
            core_tree=far_tree,
            margin=25.0,
        )
