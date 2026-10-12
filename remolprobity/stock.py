"""
Stock MolProbity on a whole model, as a remolprobity result: a reference to
compare remolprobity's results with (the oracle tests' reference).

usage: remolprobity stock MODEL -o RESULT [--workdir DIR]

molprobity.molprobity for Ramachandran, rotamers, clashes and the v33
geometry (its restraints validation); IHMValidation 3.4's
molprobity_convert.py steps for the v34 geometry. Stock defaults throughout
(NCS shortcut on), on an atom_site-only copy of the model, as IHMValidation
gives it. Only when stock's hydrogen validation fails is it run again with
that step stubbed out; only when pickling its result fails is its validation
taken from a run that does not pickle it; the provenance records either.

It runs the whole model in one process and needs about 20 KB per atom plus
1 GB of memory: only for models that stock MolProbity can take.
"""
import argparse
import math
import os
import pickle
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

from . import __version__
from . import io, memory, result
from .ids import atom_tuple, residue_id, residue_key
from .run import cctbx_commit, claim_stdout, duration, log, tool_hashes
from .tools import find_tools, versions

GB_BASE, GB_PER_ATOM = 1.0, 20e-6

# molprobity.molprobity with a validate_H that validates nothing
STUB_VALIDATE_H = '''
import sys
import mmtbx.hydrogens.validate_H as validate_H


class Skipped:
    def __init__(self, *args, **kwargs):
        pass

    def validate_inputs(self):
        pass

    def run(self):
        pass

    def get_results(self):
        return None


validate_H.validate_H = Skipped
validate_H.validate_H_results = lambda results: None
from mmtbx.command_line import molprobity
molprobity.run(sys.argv[1:])
'''

# a frame of a traceback in stock's hydrogen validation
VALIDATE_H_FRAME = re.compile(r'File "[^"]*/mmtbx/hydrogens/validate_H\.py", line \d+')


class StockError(Exception):
    """Stock MolProbity failed on the model"""


def reference(source, work):
    """Stock's result on the model at source, as a remolprobity result; its files go in work"""
    import iotbx.pdb
    import numpy as np
    from iotbx.data_manager import DataManager
    from libtbx.utils import null_out
    from mmtbx.validation.mp_validate_bonds import mp_validate_bonds

    t0 = time.time()
    budget = memory.budget()  # before the model is read, as a run does
    t = find_tools()
    log('remolprobity %s stock, with %s' % (__version__, versions(t)))
    log(budget.line)
    os.makedirs(work, exist_ok=True)
    # IHMValidation hands MolProbity an atom_site-only copy of the model: no
    # header to switch the conformation-dependent library on, no crystal
    # symmetry. remolprobity's blocks are such copies too.
    whole = io.read_model(source)
    expected = GB_BASE + GB_PER_ATOM * whole.n_atoms
    log('stock MolProbity on %d atoms: expect about %.1f GB' % (whole.n_atoms, expected))
    if budget.gb is not None and expected > budget.gb:
        log('warning: expected peak %.1f GB exceeds the memory budget of %.1f GB; stock MolProbity takes the model '
            'whole, and remolprobity in blocks needs less' % (expected, budget.gb))
    model_path = source
    if io.file_format(source) == 'cif':
        model_path = os.path.join(work, 'model.cif')
        os.replace(io.write_blocks(whole, [np.arange(len(whole.residues))], work)[0], model_path)
    prefix = os.path.join(work, 'stock')
    # as IHMValidation 3.4 runs it (ihm_validation/molprobity.py)
    args = ['disable_uc_volume_vs_n_atoms_check=True', 'coot=False', 'probe=False', 'prefix=' + prefix,
            'pickle=True', model_path]
    provenance = {'stock': True}
    for leftover in (prefix + '.out', prefix + '.pkl'):
        if os.path.exists(leftover):
            os.remove(leftover)
    run = subprocess.run(['molprobity.molprobity'] + args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         text=True)
    v = None
    if run.returncode != 0:
        error = '\n'.join(line for line in run.stderr.splitlines() if line.strip())[-2000:]
        if VALIDATE_H_FRAME.search(run.stderr):
            # Its hydrogen validation, which computes nothing compared here, can fail where the rest
            # works (1EJG: an assertion on a residue whose altlocs are different residue types); then,
            # and only then, run stock again without it, and say so
            provenance['validate_H_stubbed'] = {
                'reason': 'stock molprobity.molprobity failed in mmtbx.hydrogens.validate_H; rerun with it stubbed out',
                'error': error}
            again = subprocess.run(['mmtbx.python', '-c', STUB_VALIDATE_H] + args, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True)
            if again.returncode != 0:
                raise StockError('stock MolProbity failed, with its hydrogen validation stubbed out too:\n'
                                 + again.stderr[-4000:])
        elif os.path.exists(prefix + '.out') and os.path.exists(prefix + '.pkl'):
            # The validation completed (stock writes its .out then) and pickling it, as IHMValidation
            # asks, failed (it had opened the .pkl); stock reports only "Sorry: ...". 5PTI: an element
            # iotbx does not know stops the model's xray structure, which pickling builds. Then, and
            # only then, take stock's validation from a run in this process that does not pickle it
            from mmtbx.command_line import molprobity
            provenance['not_pickled'] = {
                'reason': "stock's validation completed, but pickling it (pickle=True, as IHMValidation 3.4 runs "
                          "it) failed; the validation is taken from molprobity.molprobity run without pickling, "
                          'in the reference process',
                'error': error}
            with open(os.devnull, 'w') as quiet:
                v, _ = molprobity.run([a if a != 'pickle=True' else 'pickle=False' for a in args], out=quiet,
                                      return_input_objects=True)
        else:
            raise StockError('stock MolProbity failed:\n' + run.stderr[-4000:])
    if v is None:
        with open(prefix + '.pkl', 'rb') as f:
            v = pickle.load(f)

    # Clashes: probe saw PDB names; map them back as molprobity_convert does.
    # Chain and residue number by residue; the residue name by the atom's own
    # (PDB) residue name, since a residue's altlocs may be different residue
    # types (microheterogeneity), as remolprobity's clashes.block_clashes does.
    names, resnames = {}, {}
    h = v.pdb_hierarchy
    if not h.fits_in_pdb_format():
        from iotbx.pdb.forward_compatible_pdb_cif_conversion import forward_compatible_pdb_cif_conversion
        renamed = h.deep_copy()
        forward_compatible_pdb_cif_conversion(hierarchy=renamed).convert_hierarchy_to_forward_compatible_pdb_representation(renamed)
        for c1, c2 in zip(h.chains(), renamed.chains()):
            for r1, r2 in zip(c1.residue_groups(), c2.residue_groups()):
                names[(c2.id.strip(), r2.resseq.strip(), r2.icode.strip())] = (
                    c1.id.strip(), residue_id(r1.resseq, r1.icode))
        for ag1, ag2 in zip(h.atom_groups(), renamed.atom_groups()):
            resnames[ag2.resname.strip()] = ag1.resname.strip()

    def named(a):
        k = (a.chain_id.strip(), a.resseq.strip(), a.icode.strip())
        if k in names:
            c, r = names[k]
            n = a.resname.strip()
            return (c, r, resnames.get(n, n), a.name.strip(), a.altloc.strip())
        return atom_tuple(a)

    pcm = v.clashes.probe_clashscore_manager
    coords = {}
    # held in a variable: atoms keep only weak links to their parents, so the
    # labels of a temporary hierarchy's atoms come out empty
    withh = iotbx.pdb.input(source_info=None, lines=pcm.h_pdb_string.splitlines()).construct_hierarchy()
    for atom in withh.atoms():
        l = atom.fetch_labels()
        coords.setdefault((l.chain_id.strip(), l.resseq.strip(), l.icode.strip(), l.resname.strip(),
                           atom.name.strip(), l.altloc.strip()), atom.xyz)

    def key(a):
        return (a.chain_id.strip(), a.resseq.strip(), a.icode.strip(), a.resname.strip(), a.name.strip(),
                a.altloc.strip())

    clashes = [(named(c.atoms_info[0]), named(c.atoms_info[1]), abs(c.overlap),
                math.dist(coords[key(c.atoms_info[0])], coords[key(c.atoms_info[1])])) for c in v.clashes.results]

    rama = [(x.chain_id.strip(), residue_id(x.resseq, x.icode), x.resname.strip(), x.altloc.strip(),
             x.ramalyze_type(), float(x.phi), float(x.psi), float(x.score)) for x in v.ramalyze.results]
    rota = [(x.chain_id.strip(), residue_id(x.resseq, x.icode), x.resname.strip(), x.altloc.strip(),
             x.evaluation, x.rotamer_name, float(x.score)) for x in v.rotalyze.results]

    def v33(kind):
        r = getattr(v.restraints, kind)
        return {'total': r.n_total, 'outliers': [tuple(atom_tuple(a) for a in o.atoms_info) +
                                                  (float(o.model), float(o.target), float(o.score)) for o in r.results]}

    # v34 as IHMValidation 3.4's molprobity_convert.py
    dm = DataManager(['model'])
    dm.process_model_file(model_path)
    m = dm.get_model()
    m.set_stop_for_unknowns(False)
    p = m.get_default_pdb_interpretation_params()
    p.pdb_interpretation.allow_polymer_cross_special_position = True
    p.pdb_interpretation.flip_symmetric_amino_acids = False
    p.pdb_interpretation.clash_guard.nonbonded_distance_threshold = None
    p.pdb_interpretation.restraints_library.cdl = True
    m.set_log(null_out())
    m.process(make_restraints=True, pdb_interpretation_params=p)
    mp = mp_validate_bonds(pdb_hierarchy=m.get_hierarchy(),
                           geometry_restraints_manager=m.get_restraints_manager().geometry, outliers_only=False)

    def no_h(x):
        return not any(a.element.strip().upper() in ('H', 'D') for a in x.atoms_info)

    residue_geometry = {}
    v34 = {}
    for slot, kind, results in ((0, 'bonds', mp.bonds.results), (2, 'angles', mp.angles.results)):
        kept = [x for x in results if no_h(x)]
        for x in kept:
            e = residue_geometry.setdefault(residue_key(x.atoms_info[0]), [0, 0.0, 0, 0.0])
            e[slot] += 1
            e[slot + 1] += float(x.score) ** 2
        observed = 'distance_value' if kind == 'bonds' else 'angle_value'
        v34[kind] = {'total': len(kept), 'outliers': [
            tuple(atom_tuple(a) for a in x.atoms_info) + (float(getattr(x, observed)), float(x.target),
                                                         float(x.sigma), float(x.score))
            for x in kept if abs(x.score) > 5]}

    model = io.read_model(model_path)
    r = result.assemble(model.residues, model.resnames, rama, rota, clashes, pcm.n_atoms,
                        {'v34': v34, 'v33': {'bonds': v33('bonds'), 'angles': v33('angles')}},
                        residue_geometry, {},
                        ramachandran_counts={k: getattr(v.ramalyze, 'n_' + k)
                                             for k in ('total', 'favored', 'allowed', 'outliers')},
                        rotamer_counts={k: getattr(v.rotalyze, 'n_' + k)
                                        for k in ('total', 'favored', 'allowed', 'outliers')})
    # stock's own clashscore counts clashes before it drops those with the same
    # id_str; kept here, outside the comparison, for oracle 1 to check. The
    # reduce, probe and het dictionary stock ran with (the installation's, as
    # remolprobity's defaults), and the cctbx commit: what the reference
    # depends on in the image
    r['provenance'] = dict(provenance, clashscore=float(v.clashes.get_clashscore()), wall_seconds=time.time() - t0,
                           tools=tool_hashes(t), cctbx=cctbx_commit(),
                           input={'path': os.path.abspath(source), 'sha256': whole.sha256, 'atoms': whole.n_atoms})
    return r


def main(argv=None):
    p = argparse.ArgumentParser(prog='remolprobity stock', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model')
    p.add_argument('-o', '--output', required=True, help="the result: .json or '-' (stdout) for JSON, else a pickle")
    p.add_argument('--workdir', help="stock's files; kept. Default: a temporary directory beside the output, "
                                     'removed on success')
    a = p.parse_args(argv)
    stdout = claim_stdout() if a.output == '-' else None
    work = a.workdir or tempfile.mkdtemp(prefix='remolprobity-stock.', dir=os.path.dirname(os.path.abspath(a.output)))
    try:
        r = reference(a.model, work)
    except (io.InputError, StockError, RuntimeError, FileNotFoundError) as e:
        log('error: %s' % e)
        if not a.workdir:
            log("stock's files are kept in %s" % work)
        return 1
    result.write(r, a.output, out=stdout)
    if not a.workdir:
        shutil.rmtree(work)
    log('done in %s: clashscore %.2f, %d Ramachandran outliers, %d rotamer outliers'
        % (duration(r['provenance']['wall_seconds']), r['clashes']['clashscore'], r['ramachandran']['outliers'],
           r['rotamers']['outliers']))
    own, child = (resource.getrusage(who).ru_maxrss / 1e6 for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    log('memory about %.2f GB without page cache: stock MolProbity %.2f GB (its largest process), this process %.2f GB'
        % (own + child, child, own))
    return 0


if __name__ == '__main__':
    sys.exit(main())
