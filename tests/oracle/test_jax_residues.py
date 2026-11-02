"""
Oracle-parity tests for JAX/GPU Ramachandran, Rotamer, C-beta, Omega, and CaBLAM
validation (remolprobity.jax_tables, remolprobity.jax_io, remolprobity.jax_residues).
"""
import glob
import os
import pickle

import numpy as np
import pytest

pytest.importorskip("jax")

from remolprobity import compare, io, jax_io, jax_residues, jax_tables, result

pytestmark = pytest.mark.oracle

CORPUS_DIR = os.environ.get("REMOLPROBITY_CORPUS", os.path.expanduser("~/.cache/remolprobity/corpus"))


def _unique_ref_files():
    by_name = {}
    for p in sorted(glob.glob(os.path.join(CORPUS_DIR, "reference", "*.pkl"))):
        base = os.path.basename(p)
        name = base.split(".")[0]
        if name not in by_name or base.endswith(".3c1d5676313508f6.pkl"):
            by_name[name] = p
    return [by_name[k] for k in sorted(by_name)]


REF_FILES = _unique_ref_files()


def test_dihedral_angles_jax():
    p0 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    p1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    p2 = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    p3 = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, -1.0]], dtype=np.float64)
    angles = np.asarray(jax_residues.dihedral_angles_jax(p0, p1, p2, p3))
    assert np.allclose(angles, [-90.0, 90.0], atol=1e-12)


def test_tables_loaded():
    tables = jax_tables.get_tables()
    assert tables["rama_tables"].shape == (6, 180, 180)
    assert len(tables["aa_list"]) in (18, 19)
    assert tables["rota_flat"].ndim == 1
    hashes = jax_tables.table_hashes()
    assert set(hashes.keys()) == {"cablam_sha256", "cdl_sha256", "probe_sha256", "rotarama_sha256"}


@pytest.mark.skipif(not REF_FILES, reason="Cached stock oracle references not found")
@pytest.mark.parametrize("ref_path", REF_FILES, ids=[os.path.basename(p).split(".")[0] for p in REF_FILES])
def test_jax_residues_against_cached_stock_oracle(ref_path):
    name = os.path.basename(ref_path).split(".")[0]
    model_path = os.path.join(CORPUS_DIR, name + ".cif")
    if not os.path.exists(model_path):
        model_path = os.path.join(CORPUS_DIR, name + ".pdb")
    if not os.path.exists(model_path):
        pytest.skip(f"Corpus model {name} not found in {CORPUS_DIR}")

    with open(ref_path, "rb") as f:
        ref = pickle.load(f)

    m = io.read_model(model_path)
    h = jax_io.read_hierarchy(model_path)
    noh = jax_io.prepare_noh_hierarchy(h)

    rama_raw = jax_residues.ramachandran(noh)
    rota_raw = jax_residues.rotamers(noh)

    index = {k: i for i, k in enumerate(m.residues)}
    rama = sorted(rama_raw, key=lambda r: (index[(r[0], r[1])], r[3], r))
    rota = sorted(rota_raw, key=lambda r: (index[(r[0], r[1])], r[3], r))
    rama_dict = dict(result.count_ramachandran(rama), records=rama)
    rota_dict = dict(result.count_rotamers(rota), records=rota)

    assert compare.differences(ref["ramachandran"], rama_dict, tol=1e-6) == []
    assert compare.differences(ref["rotamers"], rota_dict, tol=1e-6) == []

    if "cbeta" in ref:
        cb_raw = jax_residues.cbeta(noh)
        om_raw = jax_residues.omegas(noh)
        cab_rec, cab_cls = jax_residues.cablam(noh, lambda c, r: True)

        by_residue = lambda r: (index[(r[0], r[1])], r[3], r)
        om = result.filter_omega(m, om_raw)
        cab = result.filter_cablam(m, cab_rec, result.protein_runs(result.cablam_runs(m, cab_cls)))

        cb_dict = dict(result.count_cbeta(cb_raw), records=sorted(result.cbeta_outliers(cb_raw), key=by_residue))
        om_dict = dict(result.count_omega(om), records=sorted((r for r in om if r[7] != "Trans"), key=by_residue))
        cab_dict = dict(result.count_cablam(cab), records=sorted(result.cablam_flagged(cab), key=by_residue))

        assert compare.differences(ref["cbeta"], cb_dict, tol=1e-6) == []
        assert compare.differences(ref["omega"], om_dict, tol=1e-6) == []
        assert compare.differences(ref["cablam"], cab_dict, tol=1e-6) == []
