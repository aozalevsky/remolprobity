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


def nucleic_chain(n, step=7.0):
    """n residues of one chain, positions (all-atom centroids) step Å apart, each linked to the next by O3'-P"""
    pos = [np.array([step * i, 0.0, 0.0]) for i in range(n)]
    m = build([('R', str(i + 1)) for i in range(n)], pos)
    m.link_out = np.array([p + [step / 2 - 0.8, 0, 0] for p in pos])  # O3'
    m.link_in = np.array([p - [step / 2 - 0.8, 0, 0] for p in pos])  # P, 1.6 Å from the O3' before it
    return m


def test_nucleic_link_is_a_neighbour_link():
    # centroids 7 Å apart, beyond LINK_STEP, but O3' and P 1.6 Å apart: cctbx links them
    m = nucleic_chain(40)
    p = blocks.plan_blocks(m, 100, 15.0)
    b = int(p.owner[20])
    p.members[b] = np.setdiff1d(p.members[b], [21])
    p.owner[21] = (b + 1) % p.n_blocks
    with pytest.raises(blocks.PlanError, match='lacks residue'):
        blocks.check_neighbours(p, m)


def looped_chain(link):
    """
    Chain A of 30 residues; residues 11-14 (indices) a loop 80 Å away; residue 15's N link Å from
    residue 10's C, and its position 3 Å from residue 10's. One atom per residue, at its position
    """
    rng = np.random.default_rng(4)
    pos = walk(30, rng, [0.0, 0.0, 0.0])
    for i in range(11, 15):
        pos[i] = pos[i] + np.array([0.0, 0.0, 80.0])
    m = build([('A', str(i + 1)) for i in range(30)], pos, atoms_per_residue=1)
    m.link_out = np.array(pos, dtype=float) + [0.5, 0, 0]
    m.link_in = np.array(pos, dtype=float) - [0.5, 0, 0]
    m.position[15] = m.position[10] + [3.0, 0, 0]
    m.xyz[15] = m.position[15]
    m.link_in[15] = m.link_out[10] + [link, 0, 0]
    owner = np.ones(30, dtype=np.int64)
    owner[10] = 0
    return m, owner


def test_gap_whose_ends_can_link_is_filled():
    # a block holding 10 and 15 without the loop between them would join them into a peptide
    m, owner = looped_chain(1.5)
    members = blocks.block_members(m, owner, 0, 10.0)
    assert {10, 15} <= set(members) and {11, 12, 13, 14} <= set(members)


def test_gap_whose_ends_cannot_link_stays():
    m, owner = looped_chain(6.0)
    m.position[15] = m.position[10] + [6.0, 0, 0]
    m.xyz[15] = m.position[15]
    members = blocks.block_members(m, owner, 0, 10.0)
    assert {10, 15} <= set(members) and not {11, 12, 13, 14} & set(members)


def altloc_chain(tmp_path):
    """
    Chain A of 60 residues on a line, CA, C and N each; residue 2 has altlocs A then B (so the chain's
    conformers are A, B), residue 50 has B then A
    """
    lines, i = [], 0
    for r in range(1, 61):
        alts = {2: 'AB', 50: 'BA'}.get(r, ' ')
        for alt in alts:
            for name, dx in (('N', -0.5), ('CA', 0.0), ('C', 0.5)):
                i += 1
                lines.append('ATOM  %5d  %-3s%1sALA A%4d    %8.3f%8.3f%8.3f  %4.2f  0.00           %s\n'
                             % (i, name, alt.strip(), r, 3.8 * r + dx, 0, 0, 1.0 if alt == ' ' else 0.5, name[0]))
    src = tmp_path / 'alt.pdb'
    src.write_text(''.join(lines))
    return io.read_model(str(src))


def test_block_keeps_the_chains_altloc_order(tmp_path):
    m = altloc_chain(tmp_path)
    assert m.introducers == {0: [1]}
    owner = (np.arange(60) < 40).astype(np.int64)  # block 0: residues 41-60, without residue 2
    members = blocks.block_members(m, owner, 0, 10.0)
    assert 1 in members and 49 in members


def test_block_of_a_chain_without_altlocs_gets_nothing_more(tmp_path):
    m = altloc_chain(tmp_path)
    m.introducers = {}
    owner = (np.arange(60) < 40).astype(np.int64)
    assert 1 not in blocks.block_members(m, owner, 0, 10.0)


def test_rna_steps_are_linked_through_o3_and_p():
    import os
    m = io.read_model(os.path.join(os.path.dirname(__file__), '..', 'data', '1ehz_10nt.cif'))
    k = np.arange(len(m.residues) - 1)
    d = blocks.link_distance(m, k, k + 1)
    step = np.linalg.norm(np.diff(m.position, axis=0), axis=1)
    # O3'-P bonds; the last step, to 2MG, only where iotbx's residue table says it is a nucleotide
    assert np.all(d[:-1] < 1.7)
    assert np.any(step > blocks.LINK_STEP)  # all-atom centres: the old test would call some steps unlinked


def test_hairpin_strands_are_not_a_link():
    # residues 10 and 15 on two strands of a hairpin, CA 4.8 Å apart, full main chains, C and N 3.9 Å apart:
    # CaBLAM links CAs only where a residue lacks its main chain, cctbx C and N only up to 3.5 Å
    m, owner = looped_chain(3.9)
    m.position[15] = m.position[10] + [4.8, 0, 0]
    m.xyz[15] = m.position[15]
    m.has_ca = np.ones(30, dtype=bool)
    m.mainchain = np.ones(30, dtype=bool)
    members = blocks.block_members(m, owner, 0, 10.0)
    assert {10, 15} <= set(members) and not {11, 12, 13, 14} & set(members)
    # a CA trace residue whose CA is 4.3 Å away: CaBLAM links it by its CA
    m.mainchain[15] = False
    m.position[15] = m.position[10] + [4.3, 0, 0]
    m.xyz[15] = m.position[15]
    assert {11, 12, 13, 14} <= set(blocks.block_members(m, owner, 0, 10.0))
