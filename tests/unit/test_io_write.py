import numpy as np

from remolprobity import io

from test_io_cif import HEAD, ROWS, with_values
from test_io_pdb import PDB


def read(path):
    """ROWS has a '.' occupancy, there for --zero-occupancy-as-one"""
    return io.read_model(path, zero_occupancy_as_one=True)


def test_cif_blocks_round_trip(tmp_path):
    src = tmp_path / 'm.cif'
    src.write_text(HEAD + ROWS)
    m = read(str(src))
    out = tmp_path / 'blocks'
    out.mkdir()
    paths = io.write_blocks(m, [np.array([0, 2]), np.array([1])], str(out))
    assert [p.rsplit('/', 1)[1] for p in paths] == ['block00000.cif', 'block00001.cif']
    a, b = read(paths[0]), read(paths[1])
    assert a.residues == [('A', '1'), ('LONG', '10')] and a.n_atoms == 4
    assert b.residues == [('A', '2A')] and b.n_atoms == 2
    np.testing.assert_array_equal(a.xyz, m.xyz[m.res_of != 1])
    assert b.zero_occupancy == 2


def test_zero_occupancy_as_one(tmp_path):
    src = tmp_path / 'm.cif'
    src.write_text(HEAD + ROWS)
    m = read(str(src))
    out = tmp_path / 'b'
    out.mkdir()
    b = io.read_model(io.write_blocks(m, [np.array([1])], str(out), zero_occupancy_as_one=True)[0])
    assert b.zero_occupancy == 0 and b.occupancy_max == 1.0


def test_missing_b_as_zero(tmp_path):
    # no B ('.') for atom 1, an unknown one ('?') for atom 2; the others keep theirs
    src = tmp_path / 'm.cif'
    src.write_text(with_values({(1, 'B_iso_or_equiv'): '.', (2, 'B_iso_or_equiv'): '?'}))
    m = io.read_model(str(src), missing_b_as_zero=True)
    out = tmp_path / 'b'
    out.mkdir()
    everything = [np.arange(len(m.residues))]
    path = io.write_blocks(m, everything, str(out), missing_b_as_zero=True)[0]
    tags, *rows = io.cif_rows(path)
    assert [r[tags.index('B_iso_or_equiv')] for r in rows] == ['0.00', '0.00', '10.0', '10.0', '10.0', '10.0']
    assert io.read_model(path).n_atoms == 6
    # without the option, a block has what the model has
    path = io.write_blocks(m, everything, str(out), names=['as_given'])[0]
    tags, *rows = io.cif_rows(path)
    assert [r[tags.index('B_iso_or_equiv')] for r in rows][:2] == ['.', '?']


def test_pdb_blocks_keep_lines(tmp_path):
    src = tmp_path / 'm.pdb'
    src.write_text(PDB)
    m = read(str(src))
    out = tmp_path / 'b'
    out.mkdir()
    path = io.write_blocks(m, [np.array([0, 1])], str(out))[0]
    lines = open(path).read().splitlines()
    assert lines[:3] == PDB.splitlines()[1:4]
    assert lines[-1] == 'END'


def test_batched_cif_blocks_match_single_batch(tmp_path):
    src = tmp_path / 'm.cif'
    src.write_text(HEAD + ROWS)
    m = read(str(src))
    members = [np.array([0, 2]), np.array([1, 2]), np.array([2])]
    outs = {}
    for name, kw in (('one', {'max_open': 1}), ('default', {})):
        out = tmp_path / name
        out.mkdir()
        outs[name] = io.write_blocks(m, members, str(out), **kw)
    assert [p.rsplit('/', 1)[1] for p in outs['one']] == \
        ['block00000.cif', 'block00001.cif', 'block00002.cif']
    for a, b in zip(outs['one'], outs['default']):
        assert open(a, 'rb').read() == open(b, 'rb').read()


def test_batched_pdb_blocks_match_single_batch(tmp_path):
    src = tmp_path / 'm.pdb'
    src.write_text(PDB)
    m = read(str(src))
    members = [np.array([0, 1]), np.array([1, 2])]
    outs = {}
    for name, kw in (('one', {'max_open': 1}), ('default', {})):
        out = tmp_path / name
        out.mkdir()
        outs[name] = io.write_blocks(m, members, str(out), **kw)
    for a, b in zip(outs['one'], outs['default']):
        assert open(a, 'rb').read() == open(b, 'rb').read()


def test_blocks_written_under_given_names(tmp_path):
    src = tmp_path / 'm.cif'
    src.write_text(HEAD + ROWS)
    m = read(str(src))
    out = tmp_path / 'b'
    out.mkdir()
    paths = io.write_blocks(m, [np.array([1]), np.array([0])], str(out), names=['block00007.m23', 'block00002.m30'])
    assert [p.rsplit('/', 1)[1] for p in paths] == ['block00007.m23.cif', 'block00002.m30.cif']
    assert read(paths[0]).residues == [('A', '2A')] and read(paths[1]).residues == [('A', '1')]


def test_in_memory_block_text_matches_write_blocks(tmp_path):
    for ext, content in (('cif', HEAD + ROWS), ('pdb', PDB)):
        src = tmp_path / ('m.' + ext)
        src.write_text(content)
        m = read(str(src))
        members = [np.array([0, 2]), np.array([1])]
        out = tmp_path / ('b_' + ext)
        out.mkdir()
        paths = io.write_blocks(m, members, str(out), zero_occupancy_as_one=True)
        io.check_segments(m, members, paths)
        ms = io.model_slice(m, zero_occupancy_as_one=True)
        for i, (mem, p) in enumerate(zip(members, paths)):
            b_path = tmp_path / ('inmem_%d.%s' % (i, ext))
            b_path.write_text(io.block_text(ms, mem, path=p))
            m_disk, m_mem = read(p), read(str(b_path))
            assert m_disk.residues == m_mem.residues and list(m_disk.run_of) == list(m_mem.run_of)
            np.testing.assert_array_equal(m_disk.xyz, m_mem.xyz)


def test_in_memory_block_text_with_utf8_header_and_wrapped_rows(tmp_path):
    utf8_head = HEAD.replace('data_test\n', "data_test\n_struct.title 'Hämoglobin, deoxy — α₂β₂'\n")
    src = tmp_path / 'utf8.cif'
    src.write_text(utf8_head + ROWS, encoding='utf-8')
    m = io.read_model(str(src), zero_occupancy_as_one=True, missing_b_as_zero=True)
    assert m.spans is not None
    members = [np.array([0, 2]), np.array([1])]
    out = tmp_path / 'b_utf8'
    out.mkdir()
    paths = io.write_blocks(m, members, str(out), zero_occupancy_as_one=True, missing_b_as_zero=True)
    ms = io.model_slice(m, zero_occupancy_as_one=True, missing_b_as_zero=True)
    for i, (mem, p) in enumerate(zip(members, paths)):
        b_path = tmp_path / ('utf8_inmem_%d.cif' % i)
        b_path.write_text(io.block_text(ms, mem, path=p))
        assert list(io.cif_rows(str(b_path))) == list(io.cif_rows(p))



