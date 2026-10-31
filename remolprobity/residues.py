"""Ramachandran, rotamer, C-beta, omega and CaBLAM results of a hierarchy, as records"""
from .ids import residue_id


def select_chains(hierarchy, ids):
    """The first model's chains with these ids, as a hierarchy"""
    import iotbx.pdb

    ids = set(ids)
    model = hierarchy.models()[0]
    root = iotbx.pdb.hierarchy.root()
    m = iotbx.pdb.hierarchy.model(id=model.id)
    root.append_model(m)
    for chain in model.chains():
        if chain.id in ids:
            m.append_chain(chain.detached_copy())
    return root


def ramachandran(hierarchy):
    """
    Every ramalyze result as (chain, resid, resname, altloc, class, phi, psi,
    score). One chain at a time: ramalyze is quadratic in residues, and its
    residue triplets never span chains.
    """
    from mmtbx.validation.ramalyze import ramalyze
    from libtbx.utils import null_out

    chains = []
    for c in hierarchy.models()[0].chains():
        if c.id not in chains:
            chains.append(c.id)
    out = []
    for cid in chains:
        r = ramalyze(pdb_hierarchy=select_chains(hierarchy, [cid]), outliers_only=False,
                     out=null_out(), quiet=True)
        out.extend((x.chain_id.strip(), residue_id(x.resseq, x.icode), x.resname.strip(),
                    x.altloc.strip(), x.ramalyze_type(), float(x.phi), float(x.psi), float(x.score))
                   for x in r.results)
    return out


def rotamers(hierarchy):
    """Every rotalyze result as (chain, resid, resname, altloc, evaluation, name, score)"""
    from mmtbx.validation.rotalyze import rotalyze
    from libtbx.utils import null_out

    r = rotalyze(pdb_hierarchy=hierarchy, outliers_only=False, out=null_out(), quiet=True)
    return [(x.chain_id.strip(), residue_id(x.resseq, x.icode), x.resname.strip(), x.altloc.strip(),
             x.evaluation, x.rotamer_name, float(x.score)) for x in r.results]


def cbeta(hierarchy):
    """Every cbetadev result as (chain, resid, resname, altloc, deviation, dihedral NABB, occupancy)"""
    from mmtbx.validation.cbetadev import cbetadev
    from libtbx.utils import null_out

    r = cbetadev(pdb_hierarchy=hierarchy, outliers_only=False, out=null_out())
    return [(x.chain_id.strip(), residue_id(x.resseq, x.icode), x.resname.strip(), x.altloc.strip(),
             float(x.deviation), float(x.dihedral_NABB), float(x.occupancy)) for x in r.results]


def omegas(hierarchy):
    """
    Every peptide omegalyze measures, in every conformer, before its
    filter: (chain, resid, resname, altloc, previous resid, previous
    altloc, omega, type, conformer altloc, the peptide's altloc), omega
    None where it has none. omegalyze keeps a peptide without altlocs only
    from the first conformer it met after a change of chain id, which
    depends on the chain objects before it: result.filter_omega applies
    that rule over the whole model.
    """
    from mmtbx.conformation_dependent_library import generate_protein_fragments
    from mmtbx.validation.omegalyze import (find_omega_type, get_conformer_altloc, get_omega, get_omega_atoms,
                                            local_altloc_from_atoms, omega_types)

    out = []
    for twores in generate_protein_fragments(hierarchy, length=2, geometry=None, include_non_standard_peptides=True,
                                             include_d_amino_acids=True):
        atoms = get_omega_atoms(twores)
        if None in atoms:
            continue
        main = twores[1]
        omega = get_omega(atoms)
        out.append((main.parent().parent().id.strip(), residue_id(main.resseq, main.icode), main.resname.strip(),
                    local_altloc_from_atoms(atoms[2:]), residue_id(twores[0].resseq, twores[0].icode),
                    local_altloc_from_atoms(atoms[0:2]), None if omega is None else float(omega),
                    None if omega is None else omega_types[find_omega_type(omega)],
                    get_conformer_altloc(twores), local_altloc_from_atoms(atoms)))
    return out


_CABLAM = {}


def _cablam_contours():
    """CaBLAM's contour tables, read once per process"""
    if not _CABLAM:
        from mmtbx.validation import cablam
        _CABLAM.update(cablam_contours=cablam.fetch_peptide_expectations(),
                       ca_contours=cablam.fetch_ca_expectations(), motif_contours=cablam.fetch_motif_contours())
    return _CABLAM


class _AllProtein:
    """
    Within it, iotbx takes every chain and conformer for protein: cablamalyze
    then analyses a chain object of a block whatever the rest of the block
    holds of it; whether the whole chain object is protein is decided from
    all blocks (result.protein_runs)
    """

    def __enter__(self):
        import iotbx.pdb.hierarchy as hierarchy
        self.saved = [(c, c.is_protein) for c in (hierarchy.chain, hierarchy.conformer)]
        for c, _ in self.saved:
            c.is_protein = lambda self, *args, **kwds: True

    def __exit__(self, *exc):
        for c, f in self.saved:
            c.is_protein = f


def cablam(hierarchy, owned):
    """
    CaBLAM in each chain object of hierarchy on its own, as cablamalyze
    makes it: (records, classes). records: (chain, resid, resname, altloc,
    CaBLAM score, CA geometry score) of each owned residue with a CA (the
    scores None where cablamalyze has none). classes: (chain, resid, n, aa,
    na, unk) of each owned residue, 0 or 1 each, what iotbx's
    chain.is_protein counts (n: not HOH; amino acid; nucleotide; UNK), so
    that result.protein_runs can decide for the whole chain object.
    owned(chain, resid). cablamalyze keeps one chain object per chain id,
    the last; result.filter_cablam does that over the model.
    """
    import iotbx.pdb
    from iotbx.pdb import common_residue_names_get_class, residue_name_plus_atom_names_interpreter
    from libtbx.utils import null_out
    from mmtbx.validation.cablam import cablamalyze

    records, classes = [], []
    model = hierarchy.models()[0]
    for chain in model.chains():
        cid = chain.id.strip()
        keys = [(cid, residue_id(rg.resseq, rg.icode)) for rg in chain.residue_groups()]
        mine = [owned(*k) for k in keys]
        if not any(mine):
            continue
        for rg, k, own in zip(chain.residue_groups(), keys, mine):
            if not own:
                continue
            ag = rg.atom_groups()[0]
            rn = residue_name_plus_atom_names_interpreter(
                residue_name=ag.resname, atom_names=[a.name for a in ag.atoms()]).work_residue_name
            kind = None if rn is None else common_residue_names_get_class(name=rn)
            classes.append(k + (int(rn != 'HOH'), int(kind in ('common_amino_acid', 'modified_amino_acid')),
                                int(kind in ('common_rna_dna', 'modified_rna_dna')), int(rn == 'UNK')))
        root = iotbx.pdb.hierarchy.root()
        m = iotbx.pdb.hierarchy.model(id=model.id)
        root.append_model(m)
        m.append_chain(chain.detached_copy())
        with _AllProtein():
            r = cablamalyze(root, outliers_only=False, out=null_out(), quiet=True, **_cablam_contours())
        for x in r.results:
            key = (x.chain_id.strip(), residue_id(x.resseq, x.icode))
            if x.has_ca and owned(*key):
                s = x.scores
                records.append(key + (x.resname.strip(), x.altloc.strip(),
                                      None if s.cablam is None else float(s.cablam),
                                      None if s.c_alpha_geom is None else float(s.c_alpha_geom)))
    return records, classes
