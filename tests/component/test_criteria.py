"""remolprobity's C-beta, omega and v33 restraint records against stock's own analyses on one hierarchy"""
import json
import os

import pytest

from helpers import DATA, in_image

# crambin split by a TER after residue 23 into two chain objects of chain A, residue 5 (in the first) with
# altlocs A and B: stock's omegalyze then drops the second chain object's peptides without altlocs
PREP = """
import sys
src, out = sys.argv[1], sys.argv[2]
lines, prev = [], None
for line in open(src):
    if line.startswith(('ATOM', 'HETATM')):
        n = int(line[22:26])
        if n == 24 and prev == 23:
            lines.append('TER\\n')
        prev = n
        if n == 5:
            for alt, dx in (('A', 0.0), ('B', 0.4)):
                x = float(line[30:38]) + (dx if line[12:16].strip() not in ('N', 'CA', 'C', 'O') else 0.0)
                lines.append(line[:16] + alt + line[17:30] + '%8.3f' % x + line[38:54] + '  0.50' + line[60:])
            continue
        lines.append(line)
    elif line.startswith('END'):
        lines.append(line)
open(out, 'w').write(''.join(lines))
"""

DRIVER = """
import json, sys
import numpy as np
import iotbx.pdb
from libtbx.utils import null_out
from mmtbx.validation.cbetadev import cbetadev
from mmtbx.validation.omegalyze import omegalyze, omega_types
from remolprobity import io, residues, result
from remolprobity.ids import residue_id

path = sys.argv[1]
h = iotbx.pdb.input(file_name=path).construct_hierarchy()
m = io.read_model(path)

stock_cb = cbetadev(pdb_hierarchy=h, outliers_only=False, out=null_out())
ours_cb = residues.cbeta(h)
stock_cb = sorted((x.chain_id.strip(), residue_id(x.resseq, x.icode), x.resname.strip(), x.altloc.strip(),
                   round(float(x.deviation), 9)) for x in stock_cb.results)

stock_om = omegalyze(pdb_hierarchy=h, quiet=True)
stock_om = sorted((x.chain_id.strip(), residue_id(x.resseq, x.icode), x.altloc.strip(), round(float(x.omega), 9),
                   omega_types[x.omega_type]) for x in stock_om.results)
kept = result.filter_omega(m, residues.omegas(h))
ours_om = sorted((r[0], r[1], r[3], round(r[6], 9), r[7]) for r in kept)
print(json.dumps({'runs': sorted(set(m.run_of.tolist())), 'cbeta': [len(stock_cb), sorted(r[:4] + (round(r[4], 9),)
                  for r in ours_cb) == stock_cb], 'omega': [len(stock_om), len(ours_om), ours_om == stock_om]}))
"""


@pytest.mark.container
def test_cbeta_and_omega_as_stock(tmp_path):
    src = tmp_path / 'split.pdb'
    in_image('-c', PREP, os.path.join(DATA, '1crn.pdb'), src)
    out = json.loads(in_image('-c', DRIVER, src).stdout.strip().splitlines()[-1])
    assert out['runs'] == [0, 1]
    assert out['cbeta'][1], out
    # the second chain object's 22 peptides are dropped by stock, and so by filter_omega
    assert out['omega'][0] < 44 and out['omega'][2], out

CABLAM = """
import json, sys
import numpy as np
import iotbx.pdb
from libtbx.utils import null_out
from mmtbx.validation.cablam import cablamalyze
from remolprobity import io, residues, result
from remolprobity.ids import residue_id

path = sys.argv[1]
h = iotbx.pdb.input(file_name=path).construct_hierarchy()
m = io.read_model(path)
stock = cablamalyze(h, outliers_only=False, out=null_out(), quiet=True)
mine, classes = residues.cablam(h, lambda chain, resid: True)
kept = result.filter_cablam(m, mine, result.protein_runs(result.cablam_runs(m, classes)))

def norm(x):
    return None if x is None else round(x, 9)

stock_records = sorted((x.chain_id.strip(), residue_id(x.resseq, x.icode), x.altloc.strip(), norm(x.scores.cablam),
                        norm(x.scores.c_alpha_geom)) for x in stock.results if x.has_ca)
ours = sorted((r[0], r[1], r[3], norm(r[4]), norm(r[5])) for r in kept)
stock_counts = {'residues': stock.count_residues(), 'ca_residues': stock.count_ca_residues(),
                'outliers': stock.count_outliers(), 'disfavored': stock.count_disfavored(),
                'ca_outliers': stock.count_ca_outliers()}
print(json.dumps({'records': [len(stock_records), len(ours), ours == stock_records],
                  'counts': [stock_counts, result.count_cablam(kept)], 'first': stock_records[:1]}))
"""


@pytest.mark.container
@pytest.mark.parametrize('split', [True, False], ids=['split_chain', 'one_chain'])
def test_cablam_as_stock(tmp_path, split):
    src = tmp_path / 'm.pdb'
    if split:
        in_image('-c', PREP, os.path.join(DATA, '1crn.pdb'), src)
    else:
        src = os.path.join(DATA, '1crn.pdb')
    out = json.loads(in_image('-c', CABLAM, src).stdout.strip().splitlines()[-1])
    assert out['records'][2] and out['counts'][0] == out['counts'][1], out
    if split:
        # cablamalyze keeps the second chain object of chain A only, the one after the TER (residues 24-46)
        assert int(out['first'][0][1]) >= 24, out


# 4HHB with residues stock's CaBLAM scores on its 'general' contours though they are not common amino acids:
# UNK in chain A (a chain without altlocs) and in chain C, MET as MSE in chain C, and in chain C one residue
# with altlocs A and B, so that chain C takes the GPU code's path for chains with altlocs
GPU_CABLAM = """
import json, sys
import iotbx.pdb
from remolprobity import residues
try:
    from remolprobity import jax_residues
except ImportError as e:
    print(json.dumps({'skip': repr(e)})); sys.exit(0)
h = iotbx.pdb.input(file_name=sys.argv[1]).construct_hierarchy()
for chain in h.models()[0].chains():
    for rg in chain.residue_groups():
        n = int(rg.resseq)
        for ag in rg.atom_groups():
            if (chain.id.strip() == 'A' and 10 <= n <= 20) or (chain.id.strip() == 'C' and 50 <= n <= 55):
                ag.resname = 'UNK'
            elif chain.id.strip() == 'C' and ag.resname == 'MET':
                ag.resname = 'MSE'
                for a in ag.atoms():
                    if a.name.strip() == 'SD':
                        a.name, a.element = 'SE  ', 'SE'
lines = []
for line in h.as_pdb_string().splitlines(True):
    if line.startswith('ATOM') and line[21] == 'C' and int(line[22:26]) == 40 and line[12:16].strip() not in ('N', 'CA', 'C', 'O'):
        for alt, dx in (('A', 0.0), ('B', 0.4)):
            lines.append(line[:16] + alt + line[17:30] + '%8.3f' % (float(line[30:38]) + dx) + line[38:54] + '  0.50' + line[60:])
        continue
    lines.append(line)
h = iotbx.pdb.input(source_info=None, lines=''.join(lines)).construct_hierarchy()


def fresh():
    x = h.deep_copy()
    x.atoms().reset_i_seq()
    return x


own = lambda c, r: True
print(json.dumps({'cpu': residues.cablam(fresh(), own), 'gpu': jax_residues.cablam(fresh(), own)}))
"""


@pytest.mark.container
def test_gpu_cablam_scores_residues_outside_the_common_amino_acids():
    out = json.loads(in_image('-c', GPU_CABLAM, os.path.join(DATA, '4hhb.cif'),
                              env={'JAX_PLATFORMS': 'cpu', 'REMOLPROBITY_ALLOW_CPU_JAX': '1'}).stdout.strip().splitlines()[-1])
    if 'skip' in out:
        pytest.skip('jax not available in image: %s' % out['skip'])
    cpu, gpu = out['cpu'], out['gpu']
    assert cpu[1] == gpu[1]
    scored = {tuple(r[:3]) for r in cpu[0] if r[4] is not None}
    # stock scores them: the test would pass vacuously otherwise
    assert {('A', '15', 'UNK'), ('C', '52', 'UNK'), ('C', '32', 'MSE')} <= scored, sorted(scored)[:20]
    assert len(cpu[0]) == len(gpu[0])
    for c, g in zip(cpu[0], gpu[0]):
        assert c[:4] == g[:4]
        for u, v in zip(c[4:], g[4:]):
            assert (u is None) == (v is None) and (u is None or abs(u - v) < 1e-9), (c, g)
