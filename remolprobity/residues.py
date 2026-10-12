"""Ramachandran and rotamer results of a hierarchy, as records"""
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
