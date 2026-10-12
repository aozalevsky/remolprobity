"""
PDB names for reduce and probe. One table for the whole model, built from
its chain and residue names as cctbx would build it from the whole
hierarchy, so every block names chains and residues the same: reduce orders
groups by these names, and probe's atom order decides who owns a clash.
"""
from .ids import residue_id
from .io import InputError


def conversion(chain_ids, residue_names):
    """iotbx's conversion for names that don't fit PDB format, or None"""
    from iotbx.pdb.forward_compatible_pdb_cif_conversion import \
        forward_compatible_pdb_cif_conversion

    c = forward_compatible_pdb_cif_conversion(
        unique_values_dict={'chain_id': list(chain_ids), 'resname': list(residue_names)})
    return c if c.conversion_required() else None


def pdb_copy(hierarchy, conv):
    """
    A copy of hierarchy in PDB names, and names[(chain, resseq, icode)]
    (PDB names, stripped) = (chain, resid, resname) in the model's names
    """
    h = hierarchy.deep_copy()
    if conv is not None:
        conv.convert_hierarchy_to_forward_compatible_pdb_representation(h)
    if not h.fits_in_pdb_format():
        raise InputError('the block does not fit PDB format after renaming')
    names = {}
    for c1, c2 in zip(hierarchy.models()[0].chains(), h.models()[0].chains()):
        for r1, r2 in zip(c1.residue_groups(), c2.residue_groups()):
            names[(c2.id.strip(), r2.resseq.strip(), r2.icode.strip())] = (
                c1.id.strip(), residue_id(r1.resseq, r1.icode), r1.atom_groups()[0].resname.strip())
    return h, names
