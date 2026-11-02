"""
Static lookup tables, reproducible table builders, and runtime provenance for
JAX/GPU validation checks (`jax_rotarama.npz`, `jax_cablam.npz`, `jax_cdl.npz`,
`jax_probe.npz`).

All four `.npz` archives can be built directly from a MolProbity/cctbx tree via
`scripts/build_jax_tables.py` or `remolprobity.jax_tables.build_all_tables()`.
On-demand builds use `fcntl.flock` and atomic `os.replace` so concurrent worker
processes never race when populating a cold cache.
"""
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import subprocess
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _default_tables_dir():
    env_dir = os.environ.get("REMOLPROBITY_TABLES_DIR")
    if env_dir:
        return env_dir
    if os.path.isdir("/opt/remolprobity/tables"):
        return "/opt/remolprobity/tables"
    return os.path.expanduser("~/.cache/remolprobity")


def default_rotarama_path():
    return os.environ.get("REMOLPROBITY_ROTARAMA_NPZ", os.path.join(_default_tables_dir(), "jax_rotarama.npz"))


def default_cablam_path():
    return os.environ.get("REMOLPROBITY_CABLAM_NPZ", os.path.join(_default_tables_dir(), "jax_cablam.npz"))


def default_cdl_path():
    return os.environ.get("REMOLPROBITY_JAX_CDL", os.path.join(_default_tables_dir(), "jax_cdl.npz"))


def default_probe_path():
    return os.environ.get("REMOLPROBITY_PROBE_TABLES", os.path.join(_default_tables_dir(), "jax_probe.npz"))


DEFAULT_NPZ_PATH = default_rotarama_path()
DEFAULT_CABLAM_NPZ_PATH = default_cablam_path()
DEFAULT_CDL_NPZ_PATH = default_cdl_path()
DEFAULT_PROBE_NPZ_PATH = default_probe_path()

RAMA_GENERAL = 0
RAMA_GLYCINE = 1
RAMA_CISPRO = 2
RAMA_TRANSPRO = 3
RAMA_PREPRO = 4
RAMA_ILEVAL = 5

RAMA_TYPE_NAMES = (
    "General",
    "Glycine",
    "Cis-proline",
    "Trans-proline",
    "Pre-proline",
    "Isoleucine or valine",
)

COMMON_AMINO_ACIDS = frozenset((
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
))

SYM_AA = frozenset(("asp", "glu", "phe", "tyr"))

_CPU_TABLES_CACHE = None
_TABLES_CACHE = None
_CPU_CABLAM_CACHE = None
_CABLAM_CACHE = None


def _get_jax():
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        raise RuntimeError("remolprobity --gpu requires jax and jaxlib (%s)" % e) from e
    jax.config.update("jax_enable_x64", True)
    cache_dir = os.environ.get("REMOLPROBITY_JAX_CACHE_DIR")
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", cache_dir)
        except Exception:
            pass
    return jax, jnp


def lazy_jit(*jit_args, **jit_kwargs):
    """
    Defer importing jax and compiling @jax.jit until the kernel is first called.
    Injects `jax` and `jnp` into the kernel function's module globals on first call
    so importing modules in CPU worker processes does not import jax/jaxlib.
    """
    def decorator(fn):
        compiled = None

        def wrapper(*args, **kwargs):
            nonlocal compiled
            if compiled is None:
                jax_mod, jnp_mod = _get_jax()
                fn.__globals__.setdefault("jax", jax_mod)
                fn.__globals__.setdefault("jnp", jnp_mod)
                compiled = jax_mod.jit(fn, *jit_args, **jit_kwargs)
            return compiled(*args, **kwargs)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


def ensure_gpu_device(require_gpu=True):
    """
    Verify JAX runtime device and return metadata dict:
    {'jax': version, 'jaxlib': version, 'backend': backend, 'device': device_kind}.
    When require_gpu is True (the default for --gpu), raises RuntimeError if no GPU
    device is available unless REMOLPROBITY_ALLOW_CPU_JAX=1 is set.
    """
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    try:
        import jax
        import jaxlib
    except ImportError as e:
        raise RuntimeError("remolprobity --gpu requires jax and jaxlib (%s)" % e) from e
    jax.config.update("jax_enable_x64", True)
    allow_cpu = os.environ.get("REMOLPROBITY_ALLOW_CPU_JAX") == "1"
    gpu_devs = []
    try:
        gpu_devs = jax.devices("gpu")
    except RuntimeError:
        gpu_devs = []
    if gpu_devs:
        dev = gpu_devs[0]
        backend = "gpu"
    else:
        if require_gpu and not allow_cpu:
            raise RuntimeError(
                "remolprobity --gpu requires a JAX GPU device, but none is available "
                "(default backend: %s)" % jax.default_backend()
            )
        dev = jax.devices()[0]
        backend = jax.default_backend()
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": backend,
        "device": getattr(dev, "device_kind", str(dev)),
    }


_GPU_OPS = {}
_GPU_SOCK = None
_IN_GPU_SERVER = False


def register_gpu_op(name):
    """Register a GPU kernel function by name for in-process or GPU-server execution."""
    def decorator(fn):
        _GPU_OPS[name] = fn
        return fn
    return decorator


def set_gpu_sock(sock_path):
    """Configure the Unix domain socket path for dispatching GPU ops to the persistent server."""
    global _GPU_SOCK
    _GPU_SOCK = sock_path


def get_gpu_sock():
    """Return active GPU server socket path, or None if executing in-process."""
    if _IN_GPU_SERVER:
        return None
    return _GPU_SOCK or os.environ.get("REMOLPROBITY_GPU_SOCK")


def call_gpu(op_name, *args):
    """
    Execute registered GPU operation `op_name(*args)` via the persistent GPU server
    if configured, or directly in-process otherwise.
    """
    sock = get_gpu_sock()
    if sock is not None:
        from multiprocessing.connection import Client
        try:
            with Client(sock, family="AF_UNIX") as conn:
                conn.send((op_name, args))
                status, payload = conn.recv()
        except (EOFError, OSError) as e:
            raise RuntimeError("GPU server connection failed during %s: %s" % (op_name, e)) from e
        if status != "ok":
            raise RuntimeError("GPU server error in %s:\n%s" % (op_name, payload))
        return payload
    return _GPU_OPS[op_name](*args)


def _gpu_server_main(sock_path, startup_conn, require_gpu):
    """Entrypoint for the single persistent GPU server process."""
    global _IN_GPU_SERVER
    _IN_GPU_SERVER = True
    try:
        ensure_all_tables()
        gpu_info = ensure_gpu_device(require_gpu=require_gpu)
        hashes = table_hashes()
        sources = table_sources()
        get_tables()
        get_cablam_tables()
        from . import jax_clashes, jax_geometry, jax_residues  # noqa: F401
        jax_geometry.get_cdl_tables()
        jax_clashes.load_probe_tables()
        if os.path.exists(sock_path):
            os.remove(sock_path)
        from multiprocessing.connection import Listener
        listener = Listener(sock_path, family="AF_UNIX", backlog=64)
        startup_conn.send(("ok", dict(gpu_info, tables=hashes, sources=sources)))
        startup_conn.close()
    except BaseException as e:
        try:
            startup_conn.send(("error", str(e)))
            startup_conn.close()
        except Exception:
            pass
        return

    import resource
    import traceback
    try:
        while True:
            conn = listener.accept()
            try:
                req = conn.recv()
                if req is None or req[0] == "shutdown":
                    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                    conn.send(("ok", {"peak_gb": peak_gb}))
                    conn.close()
                    break
                op_name, args = req
                res = _GPU_OPS[op_name](*args)
                conn.send(("ok", res))
            except BaseException:
                try:
                    conn.send(("error", traceback.format_exc()))
                except Exception:
                    pass
            finally:
                conn.close()
    finally:
        listener.close()
        try:
            if os.path.exists(sock_path):
                os.remove(sock_path)
        except OSError:
            pass


class GPUServer:
    """
    Context manager for a single persistent GPU server process shared by all block
    workers during a `remolprobity --gpu` run. Holds the sole CUDA context and XLA
    compilation cache while block workers remain pure CPU/NumPy processes.
    """
    def __init__(self, workdir, require_gpu=True):
        self.workdir = workdir
        self.require_gpu = require_gpu
        cand = os.path.join(os.path.abspath(workdir), "gpu.sock")
        if len(cand) > 95:
            import tempfile
            fd, cand = tempfile.mkstemp(prefix="rmp_gpu_", suffix=".sock", dir="/tmp")
            os.close(fd)
            os.remove(cand)
        self.sock_path = cand
        self.proc = None
        self.info = None
        self.peak_gb = 0.0

    def start(self):
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        self.proc = ctx.Process(
            target=_gpu_server_main,
            args=(self.sock_path, send_conn, self.require_gpu),
            name="remolprobity_gpu_server",
        )
        self.proc.start()
        send_conn.close()
        try:
            status, payload = recv_conn.recv()
        except EOFError:
            self.proc.join(timeout=5)
            raise RuntimeError(
                "GPU server exited unexpectedly during startup (exit code %s)" % self.proc.exitcode
            )
        finally:
            recv_conn.close()
        if status != "ok":
            self.proc.join(timeout=5)
            raise RuntimeError(payload)
        self.info = payload
        return self

    def __enter__(self):
        return self.start()

    def stop(self):
        if self.proc is None:
            return self.peak_gb
        if self.proc.is_alive() and os.path.exists(self.sock_path):
            try:
                from multiprocessing.connection import Client
                with Client(self.sock_path, family="AF_UNIX") as conn:
                    conn.send(("shutdown", ()))
                    status, payload = conn.recv()
                    if status == "ok" and isinstance(payload, dict):
                        self.peak_gb = payload.get("peak_gb", 0.0)
            except Exception:
                pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join()
        try:
            if os.path.exists(self.sock_path):
                os.remove(self.sock_path)
        except OSError:
            pass
        self.proc = None
        return self.peak_gb

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def _resolve_cctbx_root(cctbx_root=None):
    """Locate the MolProbity/cctbx root directory containing modules/."""
    if cctbx_root and os.path.isdir(os.path.join(cctbx_root, "modules")):
        return cctbx_root
    env_root = os.environ.get("REMOLPROBITY_CCTBX_ROOT")
    if env_root and os.path.isdir(os.path.join(env_root, "modules")):
        return env_root
    try:
        import libtbx.load_env
        dist = libtbx.env.dist_path("libtbx")
        # dist is <root>/modules/cctbx_project/libtbx
        candidate = os.path.dirname(os.path.dirname(os.path.dirname(dist)))
        if os.path.isdir(os.path.join(candidate, "modules")):
            return candidate
    except Exception:
        pass
    for candidate in (
        "/opt/molprobity",
        "/home/arthur/work/validation/molprobity_container/molprobity_master_sandbox/opt/molprobity",
    ):
        if os.path.isdir(os.path.join(candidate, "modules")):
            return candidate
    raise FileNotFoundError(
        "Could not locate MolProbity/cctbx root directory (tried REMOLPROBITY_CCTBX_ROOT, "
        "libtbx.load_env, and /opt/molprobity)."
    )


def _git_head(repo_dir):
    if not os.path.isdir(repo_dir):
        return "unknown"
    try:
        r = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _probe_version_string(probe_c_path):
    if not os.path.isfile(probe_c_path):
        return "unknown"
    with open(probe_c_path) as f:
        text = f.read()
    m = re.search(r'versionString\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else "unknown"


def _source_metadata(cctbx_root):
    return {
        "cctbx": _git_head(os.path.join(cctbx_root, "modules/cctbx_project")),
        "chem_data": _git_head(os.path.join(cctbx_root, "modules/chem_data")),
        "probe_commit": _git_head(os.path.join(cctbx_root, "modules/probe")),
        "probe_version": _probe_version_string(os.path.join(cctbx_root, "modules/probe/probe.c")),
    }


def _atomic_savez(out_path, compressed=False, only_if_missing=False, **arrays):
    """
    Write an .npz archive under an exclusive file lock via a temporary file and
    atomic os.replace, preventing concurrent first-use write races across workers.
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    lock_path = out_path + ".lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if only_if_missing and os.path.isfile(out_path):
                return out_path
            tmp_path = "%s.%d.tmp.npz" % (out_path, os.getpid())
            try:
                save_fn = np.savez_compressed if compressed else np.savez
                save_fn(tmp_path, **arrays)
                os.replace(tmp_path, out_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return out_path


class _NDimTableStub:
    """Minimal unpickling target for mmtbx.rotamer.n_dim_table.NDimTable."""
    pass


class _FlexGridStub:
    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def _decode_scitbx_base256_int(buf, pos):
    b0 = buf[pos]
    length = b0 & 0x7F
    if length == 0:
        return 0, pos + 1
    val = 0
    for idx in range(pos + length - 1, pos, -1):
        val = (val << 8) + buf[idx]
    if b0 > 128:
        val = -val
    return val, pos + length


def _decode_scitbx_base256_floats(buf, dtype=np.float64):
    n, pos = _decode_scitbx_base256_int(buf, 0)
    out = np.empty(n, dtype=dtype)
    ldexp = math.ldexp
    for i in range(n):
        b0 = buf[pos]
        length = b0 & 0x7F
        if length == 0:
            out[i] = 0.0
            pos += 1
            continue
        vd = 0.0
        for idx in range(pos + length - 1, pos, -1):
            vd = (vd + buf[idx]) / 256.0
        exp, pos = _decode_scitbx_base256_int(buf, pos + length)
        val = ldexp(vd, exp)
        if b0 > 128:
            val = -val
        out[i] = val
    return out


class _FlexFloatStub:
    dtype = np.float32

    def __setstate__(self, state):
        _grid, raw = state
        if isinstance(raw, str):
            raw = raw.encode("latin1")
        self.array = _decode_scitbx_base256_floats(raw, dtype=self.dtype)

    def __iter__(self):
        return iter(self.array)

    def __len__(self):
        return len(self.array)

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)


class _FlexDoubleStub(_FlexFloatStub):
    dtype = np.float64


class _NDimTableUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "mmtbx.rotamer.n_dim_table" and name == "NDimTable":
            return _NDimTableStub
        if module == "scitbx_array_family_flex_ext":
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                if name == "grid":
                    return _FlexGridStub
                if name == "float":
                    return _FlexFloatStub
                if name == "double":
                    return _FlexDoubleStub
        return super().find_class(module, name)


def _load_ndim_pickle(path):
    with open(path, "rb") as f:
        return _NDimTableUnpickler(f, encoding="latin1").load()


def build_rotarama_npz(out_path=None, cctbx_root=None, only_if_missing=False):
    """Build jax_rotarama.npz from cctbx rama8000_tables.h and chem_data rotamer tables."""
    out_path = out_path or default_rotarama_path()
    if only_if_missing and os.path.isfile(out_path):
        return out_path
    cctbx_root = _resolve_cctbx_root(cctbx_root)
    rama_h = os.path.join(
        cctbx_root, "modules/cctbx_project/mmtbx/validation/ramachandran/rama8000_tables.h"
    )
    with open(rama_h) as f:
        text = f.read()

    names = ["general", "glycine", "cis_pro", "trans_pro", "pre_pro", "ile_val"]
    rama_tables = []
    for name in names:
        m = re.search(rf"const double linear_table_{name}\[\]\s*=\s*\{{(.*?)\}};", text, re.S)
        assert m is not None, f"Missing {name} in {rama_h}"
        vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))]
        assert len(vals) == 180 * 180
        rama_tables.append(np.array(vals, dtype=np.float64).reshape(180, 180))
    rama_tables = np.stack(rama_tables, axis=0)

    rota_dir = os.path.join(cctbx_root, "modules/chem_data/rotarama_data")
    aa_list = [
        "arg", "asn", "asp", "cys", "gln", "glu", "his", "ile", "leu",
        "lys", "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val",
    ]
    flat_tables = []
    offsets = []
    ndims = []
    min_vals = []
    max_vals = []
    n_bins = []
    do_wraps = []
    w_bins = []
    offset = 0
    for aa in aa_list:
        pkl_aa = "phetyr" if aa in ("phe", "tyr") else aa
        pkl = os.path.join(rota_dir, f"rota8000-{pkl_aa}.pickle")
        ndt = _load_ndim_pickle(pkl)
        d = ndt.nDim
        t = np.asarray(ndt.lookupTable, dtype=np.float32)
        flat_tables.append(t)
        offsets.append(offset)
        offset += len(t)
        ndims.append(d)
        pad = lambda arr, v: list(arr) + [v] * (4 - len(arr))
        min_vals.append(pad(ndt.minVal, 0.0))
        max_vals.append(pad(ndt.maxVal, 360.0))
        n_bins.append(pad(ndt.nBins, 1))
        do_wraps.append(pad([1 if w else 0 for w in ndt.doWrap], 0))
        w_bins.append(pad(ndt.wBin, 1.0))

    props_path = os.path.join(
        cctbx_root, "modules/cctbx_project/mmtbx/rotamer/sidechain_angles.props"
    )
    props = {}
    with open(props_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            props[k.strip()] = v.strip().strip('"')

    chi_atoms = {}
    for aa in aa_list + ["mse"]:
        n_chis = int(props.get(f"{aa}.chis", "0"))
        chis = []
        for i in range(1, n_chis + 1):
            atoms = [a.split(";")[0].strip() for a in props[f"{aa}.chi{i}"].split(",")[:4]]
            chis.append(atoms)
        chi_atoms[aa] = chis

    rot_names_path = os.path.join(
        cctbx_root, "modules/cctbx_project/mmtbx/rotamer/rotamer_names.props"
    )
    rot_boxes = {aa: [] for aa in aa_list + ["mse"]}
    with open(rot_names_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"')
            aa, rot_name = k.split(" ", 1)
            nums = [float(x.strip()) for x in v.replace(",", " ").split()]
            if aa in rot_boxes:
                rot_boxes[aa].append((rot_name, nums))

    source = _source_metadata(cctbx_root)
    return _atomic_savez(
        out_path,
        compressed=False,
        only_if_missing=only_if_missing,
        rama_tables=rama_tables,
        aa_list=np.array(aa_list),
        rota_flat=np.concatenate(flat_tables),
        rota_offset=np.array(offsets, dtype=np.int64),
        rota_ndim=np.array(ndims, dtype=np.int32),
        rota_min=np.array(min_vals, dtype=np.float64),
        rota_max=np.array(max_vals, dtype=np.float64),
        rota_nbins=np.array(n_bins, dtype=np.int32),
        rota_dowrap=np.array(do_wraps, dtype=np.int32),
        rota_wbin=np.array(w_bins, dtype=np.float64),
        chi_atoms_json=np.array(json.dumps(chi_atoms)),
        rot_boxes_json=np.array(json.dumps(rot_boxes)),
        source_json=np.array(json.dumps(source, sort_keys=True)),
    )


def _parse_c_string_array(header_text, array_name):
    """Extract active (non-commented) string literals from a C static const char* array."""
    # Strip C/C++ comments first
    no_block = re.sub(r"/\*.*?\*/", "", header_text, flags=re.S)
    no_comments = re.sub(r"//[^\n]*", "", no_block)
    m = re.search(rf"{array_name}\[\]\s*=\s*\{{(.*?)\}};", no_comments, re.S)
    assert m is not None, f"Could not find {array_name}[] in header"
    return re.findall(r'"([^"]*)"', m.group(1))


def _load_py_module(path, name):
    with open(path) as f:
        src = f.read()
    src = re.sub(
        r"from mmtbx\.conformation_dependent_library\.bond_angle_registry import \\\n\s*bond_angle_registry",
        "bond_angle_registry = None",
        src,
    )
    mod = type(os)(name)
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def build_cablam_npz(out_path=None, cctbx_root=None, only_if_missing=False):
    """Build jax_cablam.npz from chem_data/cablam_data and iotbx/pdb residue tables."""
    out_path = out_path or default_cablam_path()
    if only_if_missing and os.path.isfile(out_path):
        return out_path
    cctbx_root = _resolve_cctbx_root(cctbx_root)
    cablam_dir = os.path.join(cctbx_root, "modules/chem_data/cablam_data")
    pep_tables = []
    ca_tables = []
    for t in ("general", "gly", "transpro", "cispro"):
        ndt_p = _load_ndim_pickle(os.path.join(cablam_dir, f"cablam.8000.expected.{t}.pickle"))
        pep_tables.append(np.array(ndt_p.lookupTable, dtype=np.float64).reshape(ndt_p.nBins))
        ndt_c = _load_ndim_pickle(os.path.join(cablam_dir, f"cablam.8000.expected.{t}_CA.pickle"))
        ca_tables.append(np.array(ndt_c.lookupTable, dtype=np.float64).reshape(ndt_c.nBins))
    pep_tables = np.stack(pep_tables, axis=0)
    ca_tables = np.stack(ca_tables, axis=0)

    pdb_dir = os.path.join(cctbx_root, "modules/cctbx_project/iotbx/pdb")
    with open(os.path.join(pdb_dir, "common_residue_names.h")) as f:
        crn_h = f.read()
    with open(os.path.join(pdb_dir, "modified_aa_names.h")) as f:
        man_h = f.read()
    with open(os.path.join(pdb_dir, "modified_rna_dna_names.h")) as f:
        mrd_h = f.read()

    aac_mod = _load_py_module(os.path.join(pdb_dir, "amino_acid_codes.py"), "_aac")

    common_aa = np.array(_parse_c_string_array(crn_h, "amino_acid"))
    mod_aa = np.array(_parse_c_string_array(man_h, "modified_amino_acid"))
    d_aa = np.array(list(aac_mod.three_letter_l_given_three_letter_d.keys()))
    rna_dna = np.array(
        _parse_c_string_array(crn_h, "rna_dna")
        + _parse_c_string_array(crn_h, "ccp4_mon_lib_rna_dna")
    )
    mod_rna_dna = np.array(_parse_c_string_array(mrd_h, "modified_rna_dna"))
    water = np.array(_parse_c_string_array(crn_h, "water"))
    d_to_l_json = np.array(json.dumps(dict(aac_mod.three_letter_l_given_three_letter_d)))
    source = _source_metadata(cctbx_root)

    return _atomic_savez(
        out_path,
        compressed=False,
        only_if_missing=only_if_missing,
        pep_tables=pep_tables,
        ca_tables=ca_tables,
        common_aa=common_aa,
        mod_aa=mod_aa,
        d_aa=d_aa,
        rna_dna=rna_dna,
        mod_rna_dna=mod_rna_dna,
        water=water,
        d_to_l_json=d_to_l_json,
        source_json=np.array(json.dumps(source, sort_keys=True)),
    )


def build_cdl_npz(out_path=None, cctbx_root=None, only_if_missing=False):
    """Build jax_cdl.npz from mmtbx.conformation_dependent_library and iotbx.pdb."""
    out_path = out_path or default_cdl_path()
    if only_if_missing and os.path.isfile(out_path):
        return out_path
    cctbx_root = _resolve_cctbx_root(cctbx_root)
    cdl_dir = os.path.join(
        cctbx_root, "modules/cctbx_project/mmtbx/conformation_dependent_library"
    )
    pdb_dir = os.path.join(cctbx_root, "modules/cctbx_project/iotbx/pdb")
    cdl_db_mod = _load_py_module(os.path.join(cdl_dir, "cdl_database.py"), "_cdl_db")
    cdl_setup_mod = _load_py_module(os.path.join(cdl_dir, "cdl_setup.py"), "_cdl_setup")
    mod_aa_mod = _load_py_module(os.path.join(pdb_dir, "modified_aa_names.py"), "_mod_aa")
    aac_mod = _load_py_module(os.path.join(pdb_dir, "amino_acid_codes.py"), "_aac")

    cdl_database = cdl_db_mod.cdl_database
    group_names = list(cdl_database.keys())
    grid = np.full((len(group_names), 36, 36, 42), np.nan, dtype=np.float64)
    stat_type = np.zeros((len(group_names), 36, 36), dtype="U1")
    for gi, gname in enumerate(group_names):
        gdict = cdl_database[gname]
        for (phi, psi), vals in gdict.items():
            pi = (phi + 180) // 10
            si = (psi + 180) // 10
            stat_type[gi, pi, si] = vals[0]
            for k in range(1, len(vals)):
                if vals[k] is not None:
                    grid[gi, pi, si, k] = float(vals[k])

    cis_pro = [
        0, 127.0, 2.4, 102.6, 1.1, 112.1, 2.6, 112.0, 2.5, 120.2, 2.4, 117.1, 2.8, 121.1, 1.9,
        1.338, 0.019, 1.468, 0.017, 1.533, 0.018, 1.524, 0.020, 1.228, 0.020,
        120.6, 2.2, 111.5, 1.4, 103.8, 1.2, 104.0, 1.9, 105.4, 2.3,
        1.506, 0.039, 1.512, 0.027, 1.474, 0.014,
    ]
    cis_nonpro = [0, 123.0, 3.0]
    source = _source_metadata(cctbx_root)

    return _atomic_savez(
        out_path,
        compressed=True,
        only_if_missing=only_if_missing,
        group_names_json=json.dumps(group_names),
        before_pro_groups_json=json.dumps(cdl_setup_mod.before_pro_groups),
        not_before_pro_groups_json=json.dumps(cdl_setup_mod.not_before_pro_groups),
        parent_child_json=json.dumps(mod_aa_mod.lookup),
        three_letter_given_one_letter_json=json.dumps(aac_mod.three_letter_given_one_letter),
        cdl_grid=grid,
        cdl_stat_type=stat_type,
        cis_pro_eh99=np.asarray(cis_pro, dtype=np.float64),
        cis_nonpro=np.asarray(cis_nonpro, dtype=np.float64),
        source_json=np.array(json.dumps(source, sort_keys=True)),
    )


def build_probe_npz(out_path=None, cctbx_root=None, only_if_missing=False):
    """Build jax_probe.npz from modules/probe C sources and dotSphere geometry."""
    out_path = out_path or default_probe_path()
    if only_if_missing and os.path.isfile(out_path):
        return out_path
    cctbx_root = _resolve_cctbx_root(cctbx_root)
    probe_dir = os.path.join(cctbx_root, "modules/probe")

    def cpp(filename):
        return subprocess.check_output(
            ["gcc", "-E", "-I" + probe_dir, os.path.join(probe_dir, filename)],
            text=True,
        )

    ap_out = cpp("atomprops.c")
    sc_out = cpp("stdconntable.c")
    sel_out = cpp("select.c")

    enum_body = re.search(r"enum atomIdentifiers \{(.*?)\};", ap_out, re.S).group(1)
    enum_names = []
    for tok in enum_body.replace("\n", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            tok = tok.split("=")[0].strip()
        enum_names.append(tok)
    atom_id_map = {name: i for i, name in enumerate(enum_names)}
    num_types = atom_id_map["endAtomTypes"]

    tbl_body = re.search(r"atomProp AtomTbl\[\w+\] = \{(.*?)\};", ap_out, re.S).group(1)
    atno = np.zeros(num_types, dtype=np.int32)
    erad = np.zeros(num_types, dtype=np.float32)
    erad_nuc = np.zeros(num_types, dtype=np.float32)
    irad = np.zeros(num_types, dtype=np.float32)
    covrad = np.zeros(num_types, dtype=np.float32)
    elem_flags = np.zeros(num_types, dtype=np.int32)
    elem_names = ["?"] * num_types

    row_re = (
        r'\{\s*(\w+)\s*,\s*(\d+)\s*,\s*"([^"]*)"\s*,\s*([^,]+)\s*,\s*'
        r'([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*"([^"]*)"\s*,\s*([^}]+)\}'
    )
    for m in re.finditer(row_re, tbl_body):
        tname, an, nm, er, ern, ir, cr, _col, fl = m.groups()
        idx = atom_id_map[tname]
        atno[idx] = int(an)
        elem_names[idx] = nm
        erad[idx] = float(er.strip("() "))
        erad_nuc[idx] = float(ern.strip("() "))
        irad[idx] = float(ir.strip("() "))
        covrad[idx] = float(cr.strip("() "))
        elem_flags[idx] = int(eval(fl.strip()))

    sc_body = re.search(r"StandardResAtomConnRec\[\] = \{(.*?)\};", sc_out, re.S).group(1)
    stdconn = {}
    for m in re.finditer(r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"', sc_body):
        k, v = m.groups()
        if k not in stdconn:
            stdconn[k] = v

    def get_str(name):
        m = re.search(rf"char \*{name}\s*=\s*(.*?);", sel_out, re.S)
        return "".join(re.findall(r'"([^"]*)"', m.group(1)))

    def get_tbl(name):
        m = re.search(rf"ResidueAndAtomPair {name}\[\]\s*=\s*\{{(.*?)\}};", sel_out, re.S)
        rows = []
        for rm in re.finditer(r'((?:"[^"]*"\s*)+),\s*((?:"[^"]*"\s*)+),\s*([^,\n]+)', m.group(1)):
            rlist = "".join(re.findall(r'"([^"]*)"', rm.group(1)))
            alist = "".join(re.findall(r'"([^"]*)"', rm.group(2)))
            bits = int(eval(rm.group(3).strip(), dict(atom_id_map)))
            rows.append([rlist, alist, bits])
        return rows

    with open(os.path.join(probe_dir, "parse.h")) as f:
        ph = f.read()
    props_map = {}
    for m in re.finditer(r"#define\s+(\w+_PROP)\s+\((.*?)\)", ph):
        props_map[m.group(1)] = int(eval(m.group(2)))

    res_macros = {
        "HE_RESNAMES": "currently there are none RMI 070711",
        "HF_RESNAMES": ":PHF:HF3:HF5:",
        "HG_RESNAMES": ": HG:HG2:HGB:HGC:HGI:MAC:MBO:MMC:PHG:PMB:AAS:AMS:BE7:CMH:EMC:EMT:",
        "HO_RESNAMES": ": HO:HO3:",
        "HS_RESNAMES": "currently there are none RMI 070711",
    }

    pi_const = 3.14159265359

    def make_dots(radius, density=16.0):
        r_f = float(np.float32(radius))
        d_f = float(np.float32(density))
        maxpnts = int(math.floor(4.0 * pi_const * d_f * 1.0 * (r_f * r_f)))
        if maxpnts <= 0:
            return np.zeros((0, 3), dtype=np.float64)
        offset = float(np.float32(0.2))
        nequator = int(math.floor(math.sqrt(maxpnts * pi_const)))
        odd = 1
        ang = 5.0 * pi_const / 360.0
        cosang = math.cos(ang)
        sinang = math.sin(ang)
        nvert = nequator // 2
        pts = []
        for j in range(nvert + 1):
            phi = (pi_const * j) / nvert
            z0 = math.cos(phi) * r_f
            xy0 = math.sin(phi) * r_f
            nhoriz = int(math.floor(nequator * math.sin(phi)))
            if nhoriz < 1:
                nhoriz = 1
            for k in range(nhoriz):
                if odd:
                    theta = (2.0 * pi_const * k + offset) / nhoriz
                else:
                    theta = (2.0 * pi_const * k) / nhoriz
                x0 = math.cos(theta) * xy0
                y0 = math.sin(theta) * xy0
                if len(pts) >= maxpnts:
                    break
                pts.append((x0, y0 * cosang - z0 * sinang, y0 * sinang + z0 * cosang))
            odd = 0 if odd else 1
        return np.array(pts, dtype=np.float64)

    dot_spheres = {}
    for i in range(num_types):
        dot_spheres[f"dots_{i}"] = make_dots(erad[i], 16.0)
    co_rad = float(np.float32(1.70) * np.float32(1.65 / 1.70))
    dot_spheres["dots_co"] = make_dots(co_rad, 16.0)

    source = _source_metadata(cctbx_root)
    meta = {
        "atom_id_map": atom_id_map,
        "elem_names": elem_names,
        "props_map": props_map,
        "res_macros": res_macros,
        "stdconn": stdconn,
        "AAList": get_str("AAList"),
        "HphobicAAList": get_str("HphobicAAList"),
        "HphilicAAList": get_str("HphilicAAList"),
        "AromaticAAList": get_str("AromaticAAList"),
        "NAList": get_str("NAList"),
        "NAbackboneList": get_str("NAbackboneList"),
        "MethylResList": get_str("MethylResList"),
        "MethyleneResList": get_str("MethyleneResList"),
        "ChargedAAList": get_str("ChargedAAList"),
        "AlwaysChargedAtomsList": get_str("AlwaysChargedAtomsList"),
        "AmbigChargedAtomsList": get_str("AmbigChargedAtomsList"),
        "ChargedNucAcidAtomsList": get_str("ChargedNucAcidAtomsList"),
        "WaterList": get_str("WaterList"),
        "AromaticAtomsTbl": get_tbl("AromaticAtomsTbl"),
        "MethylAtomsTbl": get_tbl("MethylAtomsTbl"),
        "MethyleneAtomsTbl": get_tbl("MethyleneAtomsTbl"),
        "ChargedAAAtomsTbl": get_tbl("ChargedAAAtomsTbl"),
        "DonorAcceptorAtomTbl": get_tbl("DonorAcceptorAtomTbl"),
        "source": source,
    }

    return _atomic_savez(
        out_path,
        compressed=True,
        only_if_missing=only_if_missing,
        atno=atno,
        erad=erad,
        erad_nuc=erad_nuc,
        irad=irad,
        covrad=covrad,
        elem_flags=elem_flags,
        meta_json=np.array(json.dumps(meta)),
        source_json=np.array(json.dumps(source, sort_keys=True)),
        **dot_spheres,
    )


def build_all_tables(out_dir=None, cctbx_root=None, only_if_missing=False):
    """Build all four JAX static lookup archives in out_dir."""
    out_dir = out_dir or _default_tables_dir()
    os.makedirs(out_dir, exist_ok=True)
    return {
        "rotarama": build_rotarama_npz(os.path.join(out_dir, "jax_rotarama.npz"), cctbx_root, only_if_missing),
        "cablam": build_cablam_npz(os.path.join(out_dir, "jax_cablam.npz"), cctbx_root, only_if_missing),
        "cdl": build_cdl_npz(os.path.join(out_dir, "jax_cdl.npz"), cctbx_root, only_if_missing),
        "probe": build_probe_npz(os.path.join(out_dir, "jax_probe.npz"), cctbx_root, only_if_missing),
    }


def ensure_all_tables(cctbx_root=None):
    """Ensure all four configured .npz table archives exist on disk, building any missing ones."""
    paths = {
        "rotarama": default_rotarama_path(),
        "cablam": default_cablam_path(),
        "cdl": default_cdl_path(),
        "probe": default_probe_path(),
    }
    if not os.path.isfile(paths["rotarama"]):
        build_rotarama_npz(paths["rotarama"], cctbx_root, only_if_missing=True)
    if not os.path.isfile(paths["cablam"]):
        build_cablam_npz(paths["cablam"], cctbx_root, only_if_missing=True)
    if not os.path.isfile(paths["cdl"]):
        build_cdl_npz(paths["cdl"], cctbx_root, only_if_missing=True)
    if not os.path.isfile(paths["probe"]):
        build_probe_npz(paths["probe"], cctbx_root, only_if_missing=True)
    return paths


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_hashes(cctbx_root=None):
    """Return SHA-256 checksums of all four JAX static lookup archives."""
    paths = ensure_all_tables(cctbx_root)
    return {f"{k}_sha256": _file_sha256(p) for k, p in sorted(paths.items())}


def table_sources(cctbx_root=None):
    """Return recorded source metadata for each of the four JAX static lookup archives."""
    paths = ensure_all_tables(cctbx_root)
    out = {}
    for k, p in sorted(paths.items()):
        data = np.load(p, allow_pickle=False)
        if "source_json" in data:
            out[k] = json.loads(str(data["source_json"]))
        elif "meta_json" in data:
            meta = json.loads(str(data["meta_json"]))
            out[k] = meta.get("source", {})
        else:
            out[k] = {}
    return out


def get_cpu_tables():
    """Load and cache static Ramachandran and Rotamer metadata and NumPy arrays without importing JAX."""
    global _CPU_TABLES_CACHE
    if _CPU_TABLES_CACHE is not None:
        return _CPU_TABLES_CACHE

    npz_path = default_rotarama_path()
    if not os.path.isfile(npz_path):
        build_rotarama_npz(npz_path, only_if_missing=True)

    data = np.load(npz_path, allow_pickle=False)
    aa_list = [str(x) for x in data["aa_list"]]
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    chi_atoms_raw = json.loads(str(data["chi_atoms_json"]))
    chi_atoms = {}
    for aa, chis in chi_atoms_raw.items():
        norm_chis = []
        for c in chis:
            norm_chis.append([(x[0] if isinstance(x, list) else x).strip() for x in c])
        chi_atoms[aa] = norm_chis
    rot_boxes = json.loads(str(data["rot_boxes_json"]))

    max_boxes = max(len(v) for v in rot_boxes.values())
    n_aa = len(aa_list)
    box_low = np.full((n_aa, max_boxes, 4), -1e9, dtype=np.float64)
    box_high = np.full((n_aa, max_boxes, 4), 1e9, dtype=np.float64)
    box_valid = np.zeros((n_aa, max_boxes), dtype=bool)
    box_nchi = np.zeros((n_aa, max_boxes), dtype=np.int32)
    box_names = []
    for i, aa in enumerate(aa_list):
        names_i = []
        for j, (rname, bounds) in enumerate(rot_boxes[aa]):
            names_i.append(rname)
            box_valid[i, j] = True
            n_c = len(bounds) // 2
            box_nchi[i, j] = n_c
            for d in range(n_c):
                box_low[i, j, d] = bounds[2 * d]
                box_high[i, j, d] = bounds[2 * d + 1]
        box_names.append(names_i)

    _CPU_TABLES_CACHE = {
        "aa_list": aa_list,
        "aa_to_idx": aa_to_idx,
        "chi_atoms": chi_atoms,
        "rot_boxes": rot_boxes,
        "box_names": box_names,
        "rama_tables": np.asarray(data["rama_tables"], dtype=np.float64),
        "rota_flat": np.asarray(data["rota_flat"], dtype=np.float32),
        "rota_offset": np.asarray(data["rota_offset"], dtype=np.int64),
        "rota_ndim": np.asarray(data["rota_ndim"], dtype=np.int32),
        "rota_min": np.asarray(data["rota_min"], dtype=np.float64),
        "rota_max": np.asarray(data["rota_max"], dtype=np.float64),
        "rota_nbins": np.asarray(data["rota_nbins"], dtype=np.int32),
        "rota_dowrap": np.asarray(data["rota_dowrap"], dtype=np.int32),
        "rota_wbin": np.asarray(data["rota_wbin"], dtype=np.float64),
        "box_low": box_low,
        "box_high": box_high,
        "box_valid": box_valid,
    }
    return _CPU_TABLES_CACHE


def get_tables():
    """Load and cache static Ramachandran and Rotamer tables in JAX arrays."""
    global _TABLES_CACHE
    if _TABLES_CACHE is not None:
        return _TABLES_CACHE

    _, jnp = _get_jax()
    cpu = get_cpu_tables()
    _TABLES_CACHE = {
        "aa_list": cpu["aa_list"],
        "aa_to_idx": cpu["aa_to_idx"],
        "chi_atoms": cpu["chi_atoms"],
        "rot_boxes": cpu["rot_boxes"],
        "box_names": cpu["box_names"],
        "rama_tables": jnp.asarray(cpu["rama_tables"], dtype=jnp.float64),
        "rota_flat": jnp.asarray(cpu["rota_flat"], dtype=jnp.float32),
        "rota_offset": jnp.asarray(cpu["rota_offset"], dtype=jnp.int64),
        "rota_ndim": jnp.asarray(cpu["rota_ndim"], dtype=jnp.int32),
        "rota_min": jnp.asarray(cpu["rota_min"], dtype=jnp.float64),
        "rota_max": jnp.asarray(cpu["rota_max"], dtype=jnp.float64),
        "rota_nbins": jnp.asarray(cpu["rota_nbins"], dtype=jnp.int32),
        "rota_dowrap": jnp.asarray(cpu["rota_dowrap"], dtype=jnp.int32),
        "rota_wbin": jnp.asarray(cpu["rota_wbin"], dtype=jnp.float64),
        "box_low": jnp.asarray(cpu["box_low"], dtype=jnp.float64),
        "box_high": jnp.asarray(cpu["box_high"], dtype=jnp.float64),
        "box_valid": jnp.asarray(cpu["box_valid"], dtype=jnp.bool_),
    }
    return _TABLES_CACHE


def get_cpu_cablam_tables():
    """Load and cache static CaBLAM amino-acid/nucleic-acid class sets and NumPy tables without importing JAX."""
    global _CPU_CABLAM_CACHE
    if _CPU_CABLAM_CACHE is not None:
        return _CPU_CABLAM_CACHE

    npz_path = default_cablam_path()
    if not os.path.isfile(npz_path):
        build_cablam_npz(npz_path, only_if_missing=True)

    data = np.load(npz_path, allow_pickle=False)
    common_aa = frozenset(str(x).strip() for x in data["common_aa"])
    mod_aa = frozenset(str(x).strip() for x in data["mod_aa"])
    d_aa = frozenset(str(x).strip() for x in data["d_aa"])
    rna_dna = frozenset(str(x).strip() for x in data["rna_dna"]) if "rna_dna" in data else frozenset()
    mod_rna_dna = frozenset(str(x).strip() for x in data["mod_rna_dna"]) if "mod_rna_dna" in data else frozenset()
    d_to_l = json.loads(str(data["d_to_l_json"])) if "d_to_l_json" in data else {}
    _CPU_CABLAM_CACHE = {
        "pep_tables": np.asarray(data["pep_tables"], dtype=np.float64),
        "ca_tables": np.asarray(data["ca_tables"], dtype=np.float64),
        "common_aa": common_aa,
        "mod_aa": mod_aa,
        "d_aa": d_aa,
        "all_aa": common_aa | mod_aa | d_aa,
        "all_na": rna_dna | mod_rna_dna,
        "d_to_l": d_to_l,
    }
    return _CPU_CABLAM_CACHE


def get_cablam_tables():
    """Load and cache static CaBLAM 3D NDimTables and amino-acid class sets."""
    global _CABLAM_CACHE
    if _CABLAM_CACHE is not None:
        return _CABLAM_CACHE

    _, jnp = _get_jax()
    cpu = get_cpu_cablam_tables()
    _CABLAM_CACHE = dict(
        cpu,
        pep_tables=jnp.asarray(cpu["pep_tables"], dtype=jnp.float64),
        ca_tables=jnp.asarray(cpu["ca_tables"], dtype=jnp.float64),
    )
    return _CABLAM_CACHE
