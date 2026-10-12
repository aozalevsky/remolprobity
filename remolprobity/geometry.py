"""
Bond and angle deviations of a block, by definition:

v34 - as IHMValidation 3.4 (molprobity_convert.py) computes it on the same
      MolProbity: bonds and angles scored as mp_validate_bonds scores them -
      on cctbx from 2026-08-19, covalent-geometry and SS BOND restraints
      only; before that, every restraint - with the conformation-dependent
      library, side chains as named in the file, without hydrogens;
      outliers beyond V34_CUTOFF sigma.
v33 - as IHMValidation 3.3 (molprobity.molprobity's restraints.combined):
      no conformation-dependent library (molprobity.molprobity enables it
      only when the file says it was used in refinement), side chains
      flipped; outliers are covalent-geometry restraints beyond 4 sigma,
      totals all covalent-type restraints.

A bond or angle belongs to the residue of its first atom. Restraints are
built without the NCS shortcut, so they don't depend on what else is in
the block.
"""
import numpy as np

from .ids import atom_tuple, residue_key
from .io import InputError

DEFINITIONS = ('v34', 'v33')
V34_CUTOFF = 5.0


def _model(hierarchy, definition):
    import mmtbx.model
    from libtbx.utils import null_out

    model = mmtbx.model.manager(model_input=None, pdb_hierarchy=hierarchy, log=null_out())
    model.set_stop_for_unknowns(False)
    p = model.get_default_pdb_interpretation_params()
    pi = p.pdb_interpretation
    pi.clash_guard.nonbonded_distance_threshold = None
    pi.use_ncs_to_build_restraints = False
    if definition == 'v34':
        pi.allow_polymer_cross_special_position = True
        pi.flip_symmetric_amino_acids = False
        pi.restraints_library.cdl = True
    else:
        pi.restraints_library.cdl = False
    model.process(make_restraints=True, pdb_interpretation_params=p)
    return model


class _Owner:
    """owned(chain, resid) of atom i, remembered"""

    def __init__(self, atoms, owned):
        self.atoms, self.owned, self.labels, self.mine = atoms, owned, {}, {}

    def label(self, i):
        x = self.labels.get(i)
        if x is None:
            x = self.labels[i] = self.atoms[i].fetch_labels()
        return x

    def __call__(self, i):
        m = self.mine.get(i)
        if m is None:
            m = self.mine[i] = self.owned(*residue_key(self.label(i)))
        return m


def _v34(hierarchy, owned):
    from cctbx import geometry_restraints

    model = _model(hierarchy, 'v34')
    atoms = model.get_hierarchy().atoms()
    sites = atoms.extract_xyz()
    is_h = np.array([e.strip().upper() in ('H', 'D') for e in atoms.extract_element()], dtype=bool)
    mine = _Owner(atoms, owned)
    grm = model.get_restraints_manager().geometry
    pair = grm.pair_proxies(sites_cart=sites, flags=geometry_restraints.flags.flags(bond=True)).bond_proxies
    if pair.asu.size():
        raise InputError('symmetry-related bonds in a block')
    bonds = pair.simple
    try:  # cctbx from 2026-08-19 scores covalent geometry and SS BOND restraints only
        from mmtbx.validation.mp_validate_bonds import _scoreable_origin_ids
        scoreable = _scoreable_origin_ids()
    except ImportError:  # older cctbx scored every restraint
        scoreable = None
    out = {'residues': {}}
    for slot, kind, proxies, natoms, ideal_of in (
            (0, 'bonds', bonds, 2, lambda p: p.distance_ideal),
            (2, 'angles', grm.angle_proxies, 3, lambda p: p.angle_ideal)):
        n = proxies.size()
        iseqs = np.empty((n, natoms), dtype=np.int64)
        weight, ideal = np.empty(n), np.empty(n)
        for j, p in enumerate(proxies):
            iseqs[j] = p.i_seqs
            weight[j] = p.weight
            ideal[j] = ideal_of(p)
        delta = proxies.deltas(sites_cart=sites).as_numpy_array()
        sigma = np.sqrt(1 / weight)
        z = -delta / sigma
        keep = ~is_h[iseqs].any(axis=1)
        # An angle with the same atom twice is no angle; it could only come
        # from cctbx's old NCS shortcut, which is off, so don't hide it
        if natoms == 3:
            bad = np.flatnonzero((iseqs[:, 0] == iseqs[:, 1]) | (iseqs[:, 0] == iseqs[:, 2]) |
                                 (iseqs[:, 1] == iseqs[:, 2]))
            if bad.size:
                raise RuntimeError('degenerate angle on atoms %s' % ', '.join(
                    str(atom_tuple(mine.label(int(i)))) for i in iseqs[bad[0]]))
        if scoreable is not None:
            keep &= np.array([p.origin_id in scoreable for p in proxies], dtype=bool)
        keep &= np.array([mine(int(i)) for i in iseqs[:, 0]], dtype=bool)
        for j in np.flatnonzero(keep):
            entry = out['residues'].setdefault(residue_key(mine.label(int(iseqs[j, 0]))), [0, 0.0, 0, 0.0])
            entry[slot] += 1
            entry[slot + 1] += float(z[j] ** 2)
        outliers = []
        for j in np.flatnonzero(keep & (np.abs(z) > V34_CUTOFF)):
            outliers.append(tuple(atom_tuple(mine.label(int(i))) for i in iseqs[j]) +
                            (float(ideal[j] - delta[j]), float(ideal[j]), float(sigma[j]), float(z[j])))
        out[kind] = {'total': int(keep.sum()), 'outliers': outliers}
    return out


def _v33(hierarchy, owned):
    from cctbx import geometry_restraints
    from cctbx.geometry_restraints.auto_linking_types import iterate_covalent
    from mmtbx.validation.restraints import combined

    model = _model(hierarchy, 'v33')
    h = model.get_hierarchy()
    grm = model.get_restraints_manager().geometry
    r = combined(pdb_hierarchy=h, xray_structure=model.get_xray_structure(),
                 geometry_restraints_manager=grm, ignore_hd=True, reverse_sort=True, cdl=None)
    atoms = h.atoms()
    hd = model.get_hd_selection()
    if hd.count(True):  # restraints.combined counts without hydrogens
        grm = grm.select(~hd)
        atoms = atoms.select(~hd)
    mine = _Owner(atoms, owned)
    covalent = set(iterate_covalent())
    bonds = grm.pair_proxies(sites_cart=atoms.extract_xyz(),
                             flags=geometry_restraints.flags.flags(bond=True)).bond_proxies
    if bonds.asu.size():
        raise InputError('symmetry-related bonds in a block')

    def owned_result(o):
        return owned(*residue_key(o.atoms_info[0]))

    return {
        'bonds': {'total': sum(1 for p in bonds.simple if p.origin_id in covalent and mine(p.i_seqs[0])),
                  'outliers': [tuple(atom_tuple(a) for a in o.atoms_info) +
                               (float(o.model), float(o.target), float(o.score))
                               for o in r.bonds.results if owned_result(o)]},
        'angles': {'total': sum(1 for p in grm.angle_proxies if p.origin_id in covalent and mine(p.i_seqs[0])),
                   'outliers': [tuple(atom_tuple(a) for a in o.atoms_info) +
                                (float(o.model), float(o.target), float(o.score))
                                for o in r.angles.results if owned_result(o)]},
    }


def score(definition, hierarchy, owned):
    """The bonds and angles of hierarchy whose first atom is owned, by definition"""
    if definition == 'v34':
        return _v34(hierarchy, owned)
    if definition == 'v33':
        return _v33(hierarchy, owned)
    raise ValueError('unknown geometry definition %r (known: %s)' % (definition, ', '.join(DEFINITIONS)))
