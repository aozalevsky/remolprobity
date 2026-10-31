"""
Unit and oracle-parity tests for JAX/GPU Ramachandran and Rotamer validation
(remolprobity.jax_tables, remolprobity.jax_io, remolprobity.jax_residues).
"""
import glob
import os
import pickle
import time

import numpy as np
import pytest

from remolprobity import compare, io, jax_io, jax_residues, jax_tables, result

CORPUS_DIR = os.environ.get("REMOLPROBITY_CORPUS", os.path.expanduser("~/.cache/remolprobity/corpus"))
REF_FILES = sorted(glob.glob(os.path.join(CORPUS_DIR, "reference", "*.pkl")))


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
