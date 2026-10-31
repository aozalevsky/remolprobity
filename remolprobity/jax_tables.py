"""
Static Ramachandran and Rotamer lookup tables and metadata for JAX/GPU evaluation.
Loads pre-exported tables from ~/.cache/remolprobity/jax_rotarama.npz (or builds
them from cctbx/chem_data if missing and cctbx is importable).
"""
import json
import os
import pickle
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

DEFAULT_NPZ_PATH = os.path.expanduser("~/.cache/remolprobity/jax_rotarama.npz")

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

_TABLES_CACHE = None


def build_rotarama_npz(out_path=DEFAULT_NPZ_PATH, cctbx_root=None):
    """Build jax_rotarama.npz from cctbx rama8000_tables.h and chem_data rotamer tables."""
    import re
    if cctbx_root is None:
        cctbx_root = "/home/arthur/work/validation/molprobity_container/molprobity_master_sandbox/opt/molprobity"
    rama_h = os.path.join(
        cctbx_root, "modules/cctbx_project/mmtbx/validation/ramachandran/rama8000_tables.h"
    )
    with open(rama_h) as f:
        text = f.read()

    names = ["general", "glycine", "cis_pro", "trans_pro", "pre_pro", "ile_val"]
    rama_tables = []
    for name in names:
        m = re.search(rf"table_ramalyze_{name}\[180\]\[180\]\s*=\s*\{{(.*?)\n\s*\}};", text, re.S)
        assert m is not None, f"Missing {name} in {rama_h}"
        vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))]
        assert len(vals) == 180 * 180
        rama_tables.append(np.array(vals, dtype=np.float64).reshape(180, 180))
    rama_tables = np.stack(rama_tables, axis=0)

    rota_dir = os.path.join(cctbx_root, "modules/chem_data/rotarama_data")
    aa_list = [
        "arg", "asn", "asp", "cys", "gln", "glu", "his", "ile", "leu",
        "lys", "met", "mse", "phe", "pro", "ser", "thr", "trp", "tyr", "val",
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
        pkl = os.path.join(rota_dir, f"rota8000-{aa}.pickle")
        with open(pkl, "rb") as f:
            ndt = pickle.load(f, encoding="latin1")
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
        w_bins.append(pad(ndt.wBin, 360.0))

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
    for aa in aa_list + ["ala", "gly"]:
        n_chis = int(props.get(f"{aa}.chis", "0"))
        chis = []
        for i in range(1, n_chis + 1):
            atoms = [a.strip() for a in props[f"{aa}.chi{i}"].split(",")[:4]]
            chis.append(atoms)
        chi_atoms[aa] = chis

    rot_names_path = os.path.join(
        cctbx_root, "modules/cctbx_project/mmtbx/rotamer/rotamer_names.props"
    )
    rot_boxes = {aa: [] for aa in aa_list}
    with open(rot_names_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"')
            aa, rot_name = k.split(" ", 1)
            nums = [float(x) for x in v.split()]
            if aa in rot_boxes:
                rot_boxes[aa].append((rot_name, nums))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
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
    )
    return out_path


def get_tables():
    """Load and cache static Ramachandran and Rotamer tables in CPU/GPU arrays."""
    global _TABLES_CACHE
    if _TABLES_CACHE is not None:
        return _TABLES_CACHE

    npz_path = os.environ.get("REMOLPROBITY_ROTARAMA_NPZ", DEFAULT_NPZ_PATH)
    if not os.path.isfile(npz_path):
        build_rotarama_npz(npz_path)

    data = np.load(npz_path, allow_pickle=False)
    aa_list = [str(x) for x in data["aa_list"]]
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    chi_atoms_raw = json.loads(str(data["chi_atoms_json"]))
    chi_atoms = {}
    for aa, chis in chi_atoms_raw.items():
        norm_chis = []
        for c in chis:
            norm_chis.append([ (x[0] if isinstance(x, list) else x).strip() for x in c ])
        chi_atoms[aa] = norm_chis
    rot_boxes = json.loads(str(data["rot_boxes_json"]))

    # Pre-compile rotamer box bounds into padded numpy/JAX arrays per amino acid
    # max boxes per AA is 88 (for lys/arg), each box has up to 4 (low, high) pairs.
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

    _TABLES_CACHE = {
        "aa_list": aa_list,
        "aa_to_idx": aa_to_idx,
        "chi_atoms": chi_atoms,
        "rot_boxes": rot_boxes,
        "box_names": box_names,
        # JAX device arrays
        "rama_tables": jnp.asarray(data["rama_tables"], dtype=jnp.float64),
        "rota_flat": jnp.asarray(data["rota_flat"], dtype=jnp.float32),
        "rota_offset": jnp.asarray(data["rota_offset"], dtype=jnp.int64),
        "rota_ndim": jnp.asarray(data["rota_ndim"], dtype=jnp.int32),
        "rota_min": jnp.asarray(data["rota_min"], dtype=jnp.float64),
        "rota_max": jnp.asarray(data["rota_max"], dtype=jnp.float64),
        "rota_nbins": jnp.asarray(data["rota_nbins"], dtype=jnp.int32),
        "rota_dowrap": jnp.asarray(data["rota_dowrap"], dtype=jnp.int32),
        "rota_wbin": jnp.asarray(data["rota_wbin"], dtype=jnp.float64),
        "box_low": jnp.asarray(box_low, dtype=jnp.float64),
        "box_high": jnp.asarray(box_high, dtype=jnp.float64),
        "box_valid": jnp.asarray(box_valid, dtype=jnp.bool_),
    }
    return _TABLES_CACHE
