"""Unit tests for JAX table metadata/checksums, GPU device checks, memory budget, and GPUServer."""
import os
import sys

import numpy as np
import pytest

from remolprobity import jax_tables, run


def test_gpu_server_gb_in_resource_plan():
    assert run.main_gb(100_000, gpu=True) - run.main_gb(100_000, gpu=False) == pytest.approx(run.GPU_SERVER_GB)
    assert run.expected_gb([100_000], 1, 100_000, 10_000, gpu=True) - run.expected_gb(
        [100_000], 1, 100_000, 10_000, gpu=False
    ) == pytest.approx(run.GPU_SERVER_GB)
    sizes = [180_000] * 10
    n_cpu, _ = run.choose_workers(sizes, 1_800_000, 200_000, memory_gb=6.0, cpus=16, gpu=False)
    n_gpu, exp_gpu = run.choose_workers(sizes, 1_800_000, 200_000, memory_gb=6.0, cpus=16, gpu=True)
    assert n_gpu <= n_cpu
    assert exp_gpu <= 6.0


def test_table_hashes_and_sources_if_cached():
    paths = [
        jax_tables.default_rotarama_path(),
        jax_tables.default_cablam_path(),
        jax_tables.default_cdl_path(),
        jax_tables.default_probe_path(),
    ]
    if not all(os.path.isfile(p) for p in paths):
        pytest.skip("JAX .npz tables not built in cache")
    hashes = jax_tables.table_hashes()
    assert set(hashes) == {"rotarama_sha256", "cablam_sha256", "cdl_sha256", "probe_sha256"}
    for h in hashes.values():
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    sources = jax_tables.table_sources()
    assert set(sources) == {"rotarama", "cablam", "cdl", "probe"}
    for meta in sources.values():
        assert isinstance(meta, dict) and "cctbx" in meta and "chem_data" in meta


def test_ensure_gpu_device_rejects_cpu_without_override(monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.delenv("REMOLPROBITY_ALLOW_CPU_JAX", raising=False)
    # In a CPU-only JAX environment (or when JAX_PLATFORMS=cpu), require_gpu=True must raise RuntimeError
    import jax
    if jax.default_backend() != "gpu":
        with pytest.raises(RuntimeError, match="requires a JAX GPU device"):
            jax_tables.ensure_gpu_device(require_gpu=True)
    monkeypatch.setenv("REMOLPROBITY_ALLOW_CPU_JAX", "1")
    info = jax_tables.ensure_gpu_device(require_gpu=True)
    assert "jax" in info and "jaxlib" in info and "backend" in info and "device" in info


def test_gpu_server_rpc_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    paths = [
        jax_tables.default_rotarama_path(),
        jax_tables.default_cablam_path(),
        jax_tables.default_cdl_path(),
        jax_tables.default_probe_path(),
    ]
    if not all(os.path.isfile(p) for p in paths):
        pytest.skip("JAX .npz tables not built in cache")
    monkeypatch.setenv("REMOLPROBITY_ALLOW_CPU_JAX", "1")
    from remolprobity import jax_residues  # registers GPU ops

    with jax_tables.GPUServer(str(tmp_path), require_gpu=True) as srv:
        assert "backend" in srv.info and "tables" in srv.info
        jax_tables.set_gpu_sock(srv.sock_path)
        try:
            p0 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
            p1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
            p2 = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            p3 = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, -1.0]], dtype=np.float64)
            angles = jax_tables.call_gpu("dihedral_angles", p0, p1, p2, p3)
            assert np.allclose(angles, [-90.0, 90.0], atol=1e-12)
        finally:
            jax_tables.set_gpu_sock(None)
    assert srv.peak_gb > 0.0
