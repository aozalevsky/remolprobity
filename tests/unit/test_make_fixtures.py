import os
import sys

import numpy as np

from remolprobity import io

from test_io_cif import HEAD, ROWS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
import make_fixtures as F  # noqa: E402


def read(path):
    """The fixture as written: unchecked, since unfilled ones keep the source's '.' occupancy"""
    return io.read_model(path, check=False)


def src(tmp_path):
    p = tmp_path / 'm.cif'
    p.write_text(HEAD + ROWS)
    return str(p)


def test_excerpt_and_fill(tmp_path):
    out = str(tmp_path / 'o.cif')
    F.excerpt(src(tmp_path), out, {'A'})
    m = read(out)
    assert m.residues == [('A', '1'), ('A', '2A')] and m.n_atoms == 5 and m.zero_occupancy == 2
    F.excerpt(src(tmp_path), out, {'A'}, fill=True)
    assert read(out).zero_occupancy == 1  # '.' filled, an explicit 0.00 kept


def test_excerpt_residue_range(tmp_path):
    out = str(tmp_path / 'o.cif')
    F.excerpt(src(tmp_path), out, {'A', 'LONG'}, residues=(2, 10))
    assert read(out).residues == [('A', '2A'), ('LONG', '10')]


def test_subset_takes_whole_chains_in_order(tmp_path):
    out = str(tmp_path / 'o.cif')
    F.subset(src(tmp_path), out, 1)
    assert read(out).chain_ids == ['A']


def test_split_chain(tmp_path):
    out = str(tmp_path / 'o.cif')
    F.split_chain(src(tmp_path), out, 'A', 1)
    assert read(out).residues == [('A', '1'), ('LONG', '10'), ('A', '2A')]


def test_renumber(tmp_path):
    out = str(tmp_path / 'o.cif')
    F.renumber(src(tmp_path), out, 'A', 10000)
    assert read(out).residues == [('A', '10001'), ('A', '10002A'), ('LONG', '10')]


def test_zero_occupancy(tmp_path):
    out = str(tmp_path / 'o.cif')
    F.zero_occupancy(src(tmp_path), out, 'LONG')
    assert read(out).zero_occupancy == 3


def test_tile(tmp_path):
    out = str(tmp_path / 'o.cif')
    F.tile(src(tmp_path), out, 2, spacing=10.0)
    m, one = read(out), read(src(tmp_path))
    assert m.n_atoms == 12 and m.chain_ids == ['A_0', 'LONG_0', 'A_1', 'LONG_1']
    step = np.ptp(one.xyz, axis=0) + 10.0
    np.testing.assert_allclose(m.xyz[6:], one.xyz + [step[0], 0, 0], atol=1e-3)
