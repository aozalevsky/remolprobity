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


@pytest.fixture
def sock_path(tmp_path):
    """A socket path in tmp_path, or in /tmp if that is too long for AF_UNIX, as GPUServer does"""
    import shutil
    import tempfile

    sock = str(tmp_path / "gpu.sock")
    if len(sock) <= 95:
        yield sock
        return
    short = tempfile.mkdtemp(prefix="rmp_test_", dir="/tmp")
    try:
        yield os.path.join(short, "gpu.sock")
    finally:
        shutil.rmtree(short, ignore_errors=True)


def _serving(sock, threads, ops, monkeypatch):
    """`jax_tables._serve` on the socket sock, in a thread, with `ops` as its GPU ops"""
    import threading
    from multiprocessing.connection import Listener

    monkeypatch.setattr(jax_tables, "_GPU_OPS", dict(ops))
    listener = Listener(sock, family="AF_UNIX", backlog=64)
    server = threading.Thread(target=jax_tables._serve, args=(listener, threads), daemon=True)
    server.start()
    monkeypatch.setattr(jax_tables, "_GPU_SOCK", sock)
    return sock, server, listener


def _shutdown(sock):
    from multiprocessing.connection import Client

    with Client(sock, family="AF_UNIX") as conn:
        conn.send(("shutdown", ()))
        return conn.recv()


def test_gpu_server_runs_requests_at_once(sock_path, monkeypatch):
    # Two requests that each wait for the other: they finish only if the
    # server runs them at the same time
    import threading

    met = threading.Barrier(2, timeout=10)

    def meet(x):
        met.wait()
        return x * 2

    sock, server, listener = _serving(sock_path, 2, {"meet": meet}, monkeypatch)
    results = {}
    clients = [threading.Thread(target=lambda i=i: results.__setitem__(i, jax_tables.call_gpu("meet", i)))
               for i in (1, 2)]
    for c in clients:
        c.start()
    for c in clients:
        c.join(20)
    assert results == {1: 2, 2: 4}
    status, payload = _shutdown(sock)
    assert status == "ok" and payload["peak_gb"] > 0.0
    server.join(10)
    assert not server.is_alive()
    listener.close()


def test_gpu_server_error_is_the_requests(sock_path, monkeypatch):
    # A failing op is reported to its caller; the server goes on answering
    def fail():
        raise ValueError("no such block")

    sock, server, listener = _serving(sock_path, 2, {"fail": fail, "one": lambda: 1}, monkeypatch)
    with pytest.raises(RuntimeError, match="no such block"):
        jax_tables.call_gpu("fail")
    with pytest.raises(RuntimeError, match="KeyError"):
        jax_tables.call_gpu("missing")
    assert jax_tables.call_gpu("one") == 1
    assert _shutdown(sock)[0] == "ok"
    server.join(10)
    listener.close()


def test_gpu_server_shutdown_waits_for_running_requests(sock_path, monkeypatch):
    import threading
    import time

    started, done = threading.Event(), []

    def slow():
        started.set()
        time.sleep(0.5)
        done.append(True)
        return "slow"

    sock, server, listener = _serving(sock_path, 2, {"slow": slow}, monkeypatch)
    out = {}
    client = threading.Thread(target=lambda: out.__setitem__("r", jax_tables.call_gpu("slow")))
    client.start()
    assert started.wait(10)
    assert _shutdown(sock)[0] == "ok"
    assert done == [True]
    client.join(10)
    assert out == {"r": "slow"}
    server.join(10)
    listener.close()
