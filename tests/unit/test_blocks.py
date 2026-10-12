import numpy as np
import pytest

from remolprobity import blocks, io


def toy_model(n_chains=6, residues_per_chain=60, atoms_per_residue=8, seed=0):
    """Chains as random walks in a box: residues 3.8 Å apart, atoms around them"""
    rng = np.random.default_rng(seed)
    xyz, res_of, residues, position = [], [], [], []
    for c in range(n_chains):
        p = rng.uniform(0, 80, 3)
        for r in range(residues_per_chain):
            step = rng.normal(size=3)
            p = p + 3.8 * step / np.linalg.norm(step)
            k = len(residues)
            residues.append(('C%d' % c, str(r + 1)))
            position.append(p)
            for a in range(atoms_per_residue):
                xyz.append(p + rng.normal(scale=1.0, size=3))
                res_of.append(k)
    n = len(residues)
    return io.Model(path='toy', format='cif', sha256='0', tags=[], xyz=np.array(xyz),
                    res_of=np.array(res_of), residues=residues, resnames=['GLY'] * n,
                    position=np.array(position), chain_ids=['C%d' % c for c in range(n_chains)],
                    residue_names=['GLY'], occupancy_max=1.0, zero_occupancy=0, hydrogens=0)


def test_one_block():
    m = toy_model()
    p = blocks.plan_blocks(m, 0, 15.0)
    assert p.n_blocks == 1 and list(p.members[0]) == list(range(len(m.residues)))
    assert p.overhead(m) == 1.0


def test_every_residue_owned_once_and_cores_bounded():
    m = toy_model()
    p = blocks.plan_blocks(m, 600, 15.0)
    assert p.n_blocks > 1
    assert sorted(np.bincount(p.owner, minlength=p.n_blocks) > 0) == [True] * p.n_blocks
    n = m.atoms_per_residue()
    for b in range(p.n_blocks):
        assert n[p.owner == b].sum() <= 600
        assert set(np.flatnonzero(p.owner == b)) <= set(p.members[b])


def test_margin_holds_every_atom_near_the_core():
    m = toy_model()
    p = blocks.plan_blocks(m, 600, 15.0)
    for b in range(p.n_blocks):
        core = p.owner[m.res_of] == b
        d = np.linalg.norm(m.xyz[:, None, :] - m.xyz[core][None, :, :], axis=2).min(axis=1)
        near = np.unique(m.res_of[d < 15.0])
        assert set(near) <= set(p.members[b])


def test_plan_is_deterministic():
    m = toy_model()
    a, b = blocks.plan_blocks(m, 600, 15.0), blocks.plan_blocks(m, 600, 15.0)
    assert np.array_equal(a.owner, b.owner)
    assert all(np.array_equal(x, y) for x, y in zip(a.members, b.members))


def test_margin_below_minimum_is_refused():
    with pytest.raises(blocks.PlanError, match='minimum'):
        blocks.plan_blocks(toy_model(), 600, 9.0)


def test_missing_sequence_neighbour_is_caught():
    # residue 30 stays in block b; its neighbour 31 is moved out of b entirely
    m = toy_model()
    p = blocks.plan_blocks(m, 600, 15.0)
    b = int(p.owner[30])
    p.members[b] = np.setdiff1d(p.members[b], [31])
    p.owner[31] = (b + 1) % p.n_blocks
    with pytest.raises(blocks.PlanError, match='lacks residue'):
        blocks.check_neighbours(p, m)


def walk(n, rng, start):
    """n positions 3.8 Å apart, a random walk from start"""
    p, out = np.array(start, dtype=float), []
    for _ in range(n):
        step = rng.normal(size=3)
        p = p + 3.8 * step / np.linalg.norm(step)
        out.append(p)
    return out


def build(residues, position, atoms_per_residue=8, seed=0):
    """A model with one residue per (chain, resid) pair, atoms scattered around each position"""
    rng = np.random.default_rng(seed)
    position = np.array(position, dtype=float)
    xyz = np.repeat(position, atoms_per_residue, axis=0) + rng.normal(scale=1.0, size=(len(position) * atoms_per_residue, 3))
    res_of = np.repeat(np.arange(len(position)), atoms_per_residue)
    chains = list(dict.fromkeys(c for c, _ in residues))
    n = len(residues)
    return io.Model(path='toy', format='cif', sha256='0', tags=[], xyz=xyz, res_of=res_of,
                    residues=list(residues), resnames=['GLY'] * n, position=position,
                    chain_ids=chains, residue_names=['GLY'], occupancy_max=1.0,
                    zero_occupancy=0, hydrogens=0)


def test_unmodelled_gap_is_not_a_neighbour_break():
    # residues 30 and 31 of one chain are 60 Å apart: a gap, not a link
    rng = np.random.default_rng(1)
    pos = walk(60, rng, rng.uniform(0, 80, 3))
    pos[31:] = [q + np.array([60.0, 0.0, 0.0]) for q in pos[31:]]
    m = build([('C0', str(i + 1)) for i in range(60)], pos)
    p = blocks.plan_blocks(m, 100, 15.0)
    assert p.owner[30] != p.owner[31]


def test_ligand_with_the_chain_id_is_not_a_neighbour_break():
    # a ligand with chain C0, right after the chain's last residue, 40 Å away
    rng = np.random.default_rng(2)
    pos = walk(60, rng, rng.uniform(0, 80, 3))
    pos.append(pos[-1] + np.array([40.0, 0.0, 0.0]))
    res = [('C0', str(i + 1)) for i in range(60)] + [('C0', 'LIG')]
    m = build(res, pos)
    p = blocks.plan_blocks(m, 100, 15.0)
    assert p.owner[59] != p.owner[60]


def test_reappearing_chain_is_not_a_neighbour_break():
    # chain A, then a one-residue chain Q, then chain A again starting far away
    rng = np.random.default_rng(3)
    pos = walk(30, rng, rng.uniform(0, 80, 3))
    pos.append(pos[-1] + np.array([0.0, 30.0, 0.0]))
    pos += walk(29, rng, pos[-1] + np.array([0.0, 0.0, 60.0]))
    res = [('A', str(i + 1)) for i in range(30)] + [('Q', '1')] + \
        [('A', str(i + 31)) for i in range(29)]
    m = build(res, pos)
    p = blocks.plan_blocks(m, 200, 15.0)
    assert p.owner[29] != p.owner[31]


def test_block_members_recomputes_one_block():
    m = toy_model()
    p = blocks.plan_blocks(m, 600, 15.0)
    for b in range(p.n_blocks):
        assert np.array_equal(blocks.block_members(m, p.owner, b, 15.0), p.members[b])
    # at a larger margin: the same core, and every atom within that margin of it
    wider = blocks.block_members(m, p.owner, 1, 25.0)
    assert set(p.members[1]) < set(wider)
    core = p.owner[m.res_of] == 1
    d = np.linalg.norm(m.xyz[:, None, :] - m.xyz[core][None, :, :], axis=2).min(axis=1)
    assert set(np.unique(m.res_of[d < 25.0])) <= set(wider)
