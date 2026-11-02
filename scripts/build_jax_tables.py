#!/usr/bin/env python3
"""
Build the four static JAX lookup archives (`jax_rotarama.npz`, `jax_cablam.npz`,
`jax_cdl.npz`, `jax_probe.npz`) from a MolProbity/cctbx tree.

Usage:
    python3 scripts/build_jax_tables.py [--out-dir DIR] [--cctbx-root ROOT]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remolprobity import jax_tables


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", "--output-dir", dest="out_dir",
                   help="directory to write .npz archives (default: ~/.cache/remolprobity)")
    p.add_argument("--cctbx-root", "--molprobity-root", dest="cctbx_root",
                   help="MolProbity/cctbx root containing modules/ (default: auto-detect)")
    args = p.parse_args(argv)

    out_dir = args.out_dir or jax_tables._default_tables_dir()
    paths = jax_tables.build_all_tables(out_dir=out_dir, cctbx_root=args.cctbx_root, only_if_missing=False)
    os.environ["REMOLPROBITY_TABLES_DIR"] = out_dir
    hashes = jax_tables.table_hashes(args.cctbx_root)
    sources = jax_tables.table_sources(args.cctbx_root)
    for k, path in sorted(paths.items()):
        size = os.path.getsize(path)
        sha = hashes[f"{k}_sha256"]
        src = sources.get(k, {})
        print(f"{k:10s} {path} ({size:,d} B, sha256={sha[:16]}..., cctbx={src.get('cctbx', '?')[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
