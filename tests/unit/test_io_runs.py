"""Runs: the chain objects iotbx makes of a file, kept in block files"""
import numpy as np
import pytest

from remolprobity import io

CIF_HEAD = ('data_x\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n'
            '_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_seq_id\n'
            '_atom_site.label_asym_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n'
            '_atom_site.occupancy\n_atom_site.auth_seq_id\n_atom_site.auth_asym_id\n_atom_site.B_iso_or_equiv\n'
            '_atom_site.label_alt_id\n_atom_site.label_entity_id\n_atom_site.pdbx_PDB_model_num\n')


def cif(rows):
    """rows: (auth chain, label chain, resseq, resname)"""
    text = CIF_HEAD
    for i, (auth, label, seq, comp) in enumerate(rows):
        text += 'ATOM %d C CA %s %d %s %.3f 0.0 0.0 1.0 %d %s 10.0 . 1 1\n' % (i + 1, comp, seq, label, 3.8 * i, seq, auth)
    return text + '#\n'


def write_and_read(tmp_path, text, name, keep):
    src = tmp_path / name
    src.write_text(text)
    m = io.read_model(str(src))
    out = tmp_path / 'b'
    out.mkdir()
    path = io.write_blocks(m, [np.array(keep)], str(out))[0]
    return m, io.read_model(path)


def test_cif_chain_a_b_a_is_three_runs(tmp_path):
    src = tmp_path / 'm.cif'
    src.write_text(cif([('A', 'X', 1, 'ALA'), ('B', 'Y', 2, 'ALA'), ('A', 'X', 3, 'ALA')]))
    assert list(io.read_model(str(src)).run_of) == [0, 1, 2]


def test_cif_label_change_after_polymer_starts_a_run_but_not_after_water(tmp_path):
    src = tmp_path / 'm.cif'
    src.write_text(cif([('A', 'X', 1, 'ALA'), ('A', 'Y', 2, 'ALA'), ('A', 'Z', 3, 'HOH'),
                        ('A', 'W', 4, 'HOH')]))
    assert list(io.read_model(str(src)).run_of) == [0, 1, 2, 2]


def test_cif_block_without_b_keeps_the_break(tmp_path):
    m, b = write_and_read(tmp_path, cif([('A', 'X', 1, 'ALA'), ('B', 'Y', 2, 'ALA'), ('A', 'X', 3, 'ALA')]),
                          'm.cif', [0, 2])
    assert b.residues == [('A', '1'), ('A', '3')]
    assert list(b.run_of) == [0, 1]


def test_cif_block_keeps_names(tmp_path):
    m, b = write_and_read(tmp_path, cif([('A', 'X', 1, 'ALA'), ('B', 'Y', 2, 'ALA'), ('A', 'X', 3, 'ALA')]),
                          'm.cif', [0, 2])
    assert b.chain_ids == ['A'] and b.resnames == ['ALA', 'ALA']


def test_cif_break_is_not_invented(tmp_path):
    m, b = write_and_read(tmp_path, cif([('A', 'X', 1, 'ALA'), ('A', 'X', 2, 'ALA'), ('A', 'X', 3, 'ALA')]),
                          'm.cif', [0, 2])
    assert list(b.run_of) == [0, 0]


def test_cif_ligand_run_then_water_run_is_written(tmp_path):
    # RCSB files: a chain's ligand, then that chain's waters; iotbx merges them, which changes no result
    rows = [('A', 'X', 1, 'ALA'), ('A', 'W', 2, 'HEM'), ('B', 'Y', 3, 'ALA'), ('A', 'V', 4, 'HOH')]
    m, b = write_and_read(tmp_path, cif(rows), 'm.cif', [1, 3])
    assert b.residues == [('A', '2'), ('A', '4')] and list(b.run_of) == [0, 0]


def test_cif_polymer_after_a_ligand_run_cannot_be_kept(tmp_path):
    text = cif([('A', 'X', 1, 'ALA'), ('A', 'W', 2, 'ZN'), ('B', 'Y', 3, 'ALA'), ('A', 'X', 4, 'ALA')])
    src = tmp_path / 'm.cif'
    src.write_text(text)
    m = io.read_model(str(src))
    out = tmp_path / 'b'
    out.mkdir()
    # ZN, then the second A run: iotbx joins a polymer residue to a chain ending in a ligand
    with pytest.raises(io.InputError, match='chain break'):
        io.write_blocks(m, [np.array([1, 3])], str(out))


def test_pdb_ter_and_chain_runs(tmp_path):
    def atom(i, chain, res):
        return 'ATOM  %5d  CA  ALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n' % (i, chain, res, 3.8 * i, 0, 0)
    text = atom(1, 'A', 1) + atom(2, 'A', 2) + 'TER\n' + atom(3, 'A', 3) + atom(4, 'B', 4) + atom(5, 'A', 5)
    src = tmp_path / 'm.pdb'
    src.write_text(text)
    m = io.read_model(str(src))
    assert list(m.run_of) == [0, 0, 1, 2, 3]
    out = tmp_path / 'b'
    out.mkdir()
    path = io.write_blocks(m, [np.array([0, 2, 4])], str(out))[0]
    assert 'TER' in open(path).read()
    assert list(io.read_model(path).run_of) == [0, 1, 2]


def test_stretches_are_runs_of_one_chain_id(tmp_path):
    src = tmp_path / 'm.cif'
    # the label change splits a chain object (run) but not a stretch, reduce's unit
    src.write_text(cif([('A', 'X', 1, 'ALA'), ('A', 'Y', 2, 'ALA'), ('B', 'Z', 3, 'ALA'), ('A', 'X', 4, 'ALA')]))
    m = io.read_model(str(src))
    assert list(io.stretches(m)) == [0, 0, 1, 2]
    assert list(m.run_of) == [0, 1, 2, 3]


def test_rcsb_layout_blocks_are_written(tmp_path):
    # 4HHB: polymer chains, then each chain's HEM, then each chain's HOH; small blocks at the
    # minimum margin put a chain's ligand and waters next to each other
    import os
    from remolprobity import blocks
    m = io.read_model(os.path.join(os.path.dirname(__file__), '..', 'data', '4hhb.cif'))
    plan = blocks.plan_blocks(m, 500, 15.0)
    io.write_blocks(m, plan.members, str(tmp_path))
