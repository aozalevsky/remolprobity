"""
Planning blocks: cores by k-d median splits of residue positions, margins by
distance. Every atom within the margin of a core is in its block, whole
residues always; with the margin at least MIN_MARGIN, that holds every atom
cctbx may link to an owned atom (3 Å) and every contact probe scores.
"""
from dataclasses import dataclass

import numpy as np

MIN_MARGIN = 10.0  # Å: sequence neighbours i±2 and linked atoms are within it
LINK_STEP = 5.0  # Å between linked residues' positions: CA-CA is 3.8 (trans) or 2.9 (cis); farther pairs are not linked (a gap, a ligand or water with the chain's id)


class PlanError(Exception):
    """A plan that could not give whole-model results"""


@dataclass
class Plan:
    """owner[r]: the block whose core holds residue r; members[b]: block b's residues, core and margin"""
    owner: np.ndarray
    members: list
    margin: float
    block_atoms: int

    @property
    def n_blocks(self):
        return len(self.members)

    def sizes(self, model):
        n = model.atoms_per_residue()
        return [int(n[m].sum()) for m in self.members]

    def overhead(self, model):
        return sum(self.sizes(model)) / model.n_atoms


def _split(position, n_atoms, idx, block_atoms):
    """Halve along the longest side, at the atom-weighted median, until cores fit"""
    if n_atoms[idx].sum() <= block_atoms or len(idx) < 2:
        return [np.sort(idx)]
    pts = position[idx]
    axis = int(np.argmax(np.ptp(pts, axis=0)))
    order = idx[np.argsort(pts[:, axis], kind='stable')]
    cum = np.cumsum(n_atoms[order])
    half = int(np.clip(np.searchsorted(cum, cum[-1] / 2), 1, len(order) - 1))
    return _split(position, n_atoms, order[:half], block_atoms) + \
        _split(position, n_atoms, order[half:], block_atoms)


def block_members(model, owner, block, margin, atom_owner=None):
    """
    Residue indices of a block: its core (owner[r] == block) and every
    residue with an atom within margin of the core. atom_owner, the block
    of each atom (owner[model.res_of]), may be given to save its making
    """
    from scipy.spatial import cKDTree

    if atom_owner is None:
        atom_owner = owner[model.res_of]
    idx = np.flatnonzero(owner == block)
    core = atom_owner == block
    cxyz = model.xyz[core]
    lo, hi = cxyz.min(axis=0) - margin, cxyz.max(axis=0) + margin
    cand = np.flatnonzero(np.all((model.xyz >= lo) & (model.xyz <= hi), axis=1) & ~core)
    d, _ = cKDTree(cxyz).query(model.xyz[cand], distance_upper_bound=margin)
    near = np.unique(model.res_of[cand[np.isfinite(d)]])
    return np.union1d(idx, near)


def plan_blocks(model, block_atoms, margin):
    """
    Cores of at most block_atoms atoms (all residues in one block when
    block_atoms is 0 or the model is smaller), each with every residue that
    has an atom within margin of the core
    """
    if margin < MIN_MARGIN:
        raise PlanError('margin %.1f Å is below the minimum of %.1f Å' % (margin, MIN_MARGIN))
    n_atoms = model.atoms_per_residue()
    nres = len(model.residues)
    if block_atoms <= 0 or model.n_atoms <= block_atoms:
        return Plan(np.zeros(nres, dtype=np.int64), [np.arange(nres)], margin, block_atoms)
    cores = _split(model.position, n_atoms, np.arange(nres), block_atoms)
    owner = np.empty(nres, dtype=np.int64)
    for b, idx in enumerate(cores):
        owner[idx] = b
    atom_owner = owner[model.res_of]
    members = [block_members(model, owner, b, margin, atom_owner) for b in range(len(cores))]
    plan = Plan(owner, members, margin, block_atoms)
    check_neighbours(plan, model)
    return plan


def check_neighbours(plan, model, reach=2):
    """
    Every owned residue's sequence neighbours are in its block. A sequence
    neighbour is a residue up to reach positions away in file order, reached
    through an unbroken run of linked residues: consecutive residues of one
    chain object (iotbx's, model.run_of; else one chain id) whose positions
    are at most LINK_STEP apart. Gaps, ligands and
    waters that carry the chain's id, and a chain id that reappears after
    another chain, are not linked, so they are not neighbours. Ramachandran
    and the conformation-dependent library need the linked residues i-2..i+2.
    With reach * LINK_STEP <= MIN_MARGIN, every such neighbour of a residue
    placed by its CA is within the margin, so this check guards the margin
    code and does not refuse a correct plan.
    """
    nres = len(model.residues)
    if nres < 2:
        return
    if model.run_of is not None:
        same = model.run_of[:-1] == model.run_of[1:]  # iotbx's chain objects
    else:
        chain = [c for c, _ in model.residues]
        same = np.fromiter((chain[k] == chain[k + 1] for k in range(nres - 1)), dtype=bool, count=nres - 1)
    step = np.linalg.norm(np.diff(model.position, axis=0), axis=1)
    linked = same & (step <= LINK_STEP)

    # need[(sign, d)][r]: residue r has the neighbour r + sign * d
    need = {}
    run = np.ones(nres, dtype=bool)
    for d in range(1, reach + 1):
        m = nres - d
        if m <= 0:
            break
        run = run[:m] & linked[d - 1:d - 1 + m]  # linked[r..r+d-1]
        need[(1, d)] = np.concatenate([run, np.zeros(d, dtype=bool)])
    run = np.ones(nres, dtype=bool)
    for d in range(1, reach + 1):
        if d >= nres:
            break
        new = np.zeros(nres, dtype=bool)
        new[d:] = run[d:] & linked[0:nres - d]  # linked[r-d..r-1]
        run = new
        need[(-1, d)] = run

    owner = plan.owner
    member_keys = np.sort(np.concatenate(
        [b * nres + np.asarray(m, dtype=np.int64) for b, m in enumerate(plan.members)]))
    for (sign, d), mask in need.items():
        r = np.flatnonzero(mask)
        j = r + sign * d
        keys = owner[r].astype(np.int64) * nres + j
        pos = np.searchsorted(member_keys, keys)
        found = (pos < len(member_keys)) & (member_keys[np.minimum(pos, len(member_keys) - 1)] == keys)
        bad = np.flatnonzero(~found)
        if len(bad):
            i = bad[0]
            b = int(owner[r[i]])
            raise PlanError('block %d lacks residue %s %s next to %s %s; use a larger margin'
                            % ((b,) + model.residues[j[i]] + model.residues[r[i]]))
