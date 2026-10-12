"""One block, in its own process: everything its core owns, as plain data"""
import os
import pickle
import resource
import shutil
import time

from . import clashes, geometry, residues
from .ids import residue_id, residue_key
from .io import InputError

_SHARED = {}


def _shared(path):
    if _SHARED.get('path') != path:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        _SHARED.clear()
        _SHARED.update(data, path=path)
    return _SHARED


def _hd_selection(h):
    """
    Hydrogen and deuterium atoms. Stock takes the scattering type from the
    monomer library when the element column is blank; there, the atom name
    stands in (an ion named like its residue, HG in HG, is not hydrogen).
    """
    from scitbx.array_family import flex
    sel = flex.bool(h.atoms_size(), False)
    for i, a in enumerate(h.atoms()):
        e = a.element.strip()
        if e:
            sel[i] = e in ('H', 'D')
        else:
            name = a.name.strip().lstrip('0123456789')
            sel[i] = name[:1] in ('H', 'D') and a.fetch_labels().resname.strip() != name
    return sel


def run_block(task):
    """
    task: {'block', 'path' (the block's file), 'workdir', 'shared' (a pickle
    of owner {(chain, resid): block}, conversion, occupancy_max, tools,
    definitions, margin), optionally 'margin' (the block's own, Å)}. A
    block whose margin is too small for its core returns {'block',
    'margin', 'margin_needed', 'message'} instead of its results, for the
    run to plan it again with that margin.
    """
    import iotbx.pdb
    from scipy.spatial import cKDTree

    t0 = time.time()
    s = _shared(task['shared'])
    b = task['block']
    owner = s['owner']

    def owned(chain, resid):
        o = owner.get((chain, resid))
        if o is None:
            raise KeyError('residue %s %s of block %d is not in the model' % (chain, resid, b))
        return o == b

    workdir = os.path.join(task['workdir'], 'scratch', 'b%05d' % b)
    os.makedirs(workdir, exist_ok=True)
    h = iotbx.pdb.input(file_name=task['path']).construct_hierarchy()
    try:
        # stock MolProbity stops on these too, in pdb_interpretation; here
        # before anything is computed
        h.overall_counts().raise_duplicate_atom_labels_if_necessary()
    except Exception as e:
        raise InputError('duplicate atoms (same residue, name and altloc): %s' % e)
    h.atoms().reset_i_seq()
    owned_keys, seen = [], set()
    for chain in h.models()[0].chains():
        for rg in chain.residue_groups():
            key = (chain.id.strip(), residue_id(rg.resseq, rg.icode))
            if owned(*key) and key not in seen:
                seen.add(key)
                owned_keys.append(key)

    # as molprobity.molprobity: without hydrogens (the element, else, as
    # stock's scattering type does, the atom name), a copy that shares no atoms
    # with h, atoms that appear later in a residue merged into it, and
    # symmetric side chains named as pdb_interpretation names them
    hd = _hd_selection(h)
    noh = h.select(~hd, copy_atoms=True) if hd.count(True) else h.deep_copy()
    noh.merge_atoms_at_end_to_residues()
    noh.atoms().reset_i_seq()
    noh.flip_symmetric_amino_acids()
    times = {}
    t = time.time()
    rama = [r for r in residues.ramachandran(noh) if owned(r[0], r[1])]
    rota = [r for r in residues.rotamers(noh) if owned(r[0], r[1])]
    times['ramachandran_rotamers'] = time.time() - t

    core = [a.xyz for a in noh.atoms() if owned(*residue_key(a.fetch_labels()))]
    margin = task.get('margin', s['margin'])
    t = time.time()
    try:
        clash = clashes.block_clashes(noh, s['conversion'], owned, cKDTree(core) if core else None,
                                      margin, s['tools'], workdir, s['occupancy_max'], s.get('stretch'))
    except clashes.MarginError as e:
        if e.needed is None:
            raise
        shutil.rmtree(workdir, ignore_errors=True)
        return {'block': b, 'margin': margin, 'margin_needed': e.needed, 'message': str(e)}
    times['clashes'] = time.time() - t
    del noh

    geo = {}
    for d in s['definitions']:
        t = time.time()
        geo[d] = geometry.score(d, h.deep_copy(), owned)
        times['geometry_' + d] = time.time() - t
    os.rmdir(workdir)
    times['total'] = time.time() - t0
    return {'block': b, 'atoms': h.atoms_size(), 'owned': owned_keys, 'ramachandran': rama,
            'rotamers': rota, 'clashes': clash['records'], 'probe_atoms': clash['probe_atoms'],
            'geometry': geo, 'times': times,
            'peak_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
            # the largest of its reduce and probe processes
            'children_peak_gb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6}
