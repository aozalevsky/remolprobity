import ast
import json
import os
import shutil

import pytest

from helpers import DATA, in_image

DRIVER = """
import json, os, pickle, sys, tempfile
import numpy as np
from remolprobity import io, naming, tools, worker, geometry
path = sys.argv[1]
m = io.read_model(path)
d = tempfile.mkdtemp()
block = io.write_blocks(m, [np.arange(len(m.residues))], d)[0]
shared = os.path.join(d, 'shared.pkl')
with open(shared, 'wb') as f:
    pickle.dump({'owner': {k: 0 for k in m.residues}, 'conversion': naming.conversion(m.chain_ids, m.residue_names),
                 'occupancy_max': m.occupancy_max, 'tools': tools.find_tools(),
                 'definitions': list(geometry.DEFINITIONS), 'margin': 15.0}, f)
try:
    r = worker.run_block({'block': 0, 'path': block, 'workdir': d, 'shared': shared})
    print(json.dumps({'owned': len(r['owned']), 'residues': len(m.residues), 'rama': len(r['ramachandran']),
                      'clashes': len(r['clashes']), 'probe_atoms': r['probe_atoms'],
                      'v34': r['geometry']['v34']['bonds']['total'], 'v33': r['geometry']['v33']['bonds']['total']}))
except Exception as e:
    print(json.dumps({'error': '%s: %s' % (type(e).__name__, e)}))
"""


@pytest.mark.container
def test_one_block_of_crambin():
    out = json.loads(in_image('-c', DRIVER, os.path.join(DATA, '1crn.pdb')).stdout.strip().splitlines()[-1])
    assert out['owned'] == out['residues'] == 46
    # 327 atoms in the file; probe counts them with reduce's hydrogens
    assert out['rama'] > 40 and out['clashes'] == 0 and out['probe_atoms'] > 327
    assert out['v34'] > 300 and out['v33'] > 300


@pytest.mark.container
def test_duplicate_atoms_fail(tmp_path):
    # Review Focus 5: a duplicated atom is an error, not dropped
    src = open(os.path.join(DATA, '1crn.pdb')).read().splitlines(True)
    first = next(i for i, l in enumerate(src) if l.startswith('ATOM'))
    p = tmp_path / 'dup.pdb'
    p.write_text(''.join(src[:first + 1] + [src[first]] + src[first + 1:]))
    out = json.loads(in_image('-c', DRIVER, str(p)).stdout.strip().splitlines()[-1])
    assert out.get('error', '').startswith('InputError: duplicate atoms (same residue, name and altloc)')


PREP = """
import os, sys
import iotbx.pdb
from scitbx import matrix
from remolprobity import tools

src, out = sys.argv[1], sys.argv[2]
h = iotbx.pdb.input(file_name=src).construct_hierarchy()
for rg in h.models()[0].chains()[0].residue_groups():
    ag = rg.atom_groups()[0]
    if int(rg.resseq) == 10:
        a, b = ag.get_atom('NH1'), ag.get_atom('NH2')
        a.xyz, b.xyz = b.xyz, a.xyz
    if int(rg.resseq) == 43:
        # a flip pdb_interpretation makes, and a bond stretched to an outlier
        a, b = ag.get_atom('OD1'), ag.get_atom('OD2')
        a.xyz, b.xyz = b.xyz, a.xyz
        cg = matrix.col(ag.get_atom('CG').xyz)
        v = matrix.col(a.xyz) - cg
        a.xyz = tuple(cg + v.normalize() * (abs(v) + 0.25))
# altlocs, whose hydrogens reduce places per conformer
for rg in h.models()[0].chains()[0].residue_groups():
    if int(rg.resseq) in (11, 13, 17):
        ag = rg.atom_groups()[0]
        b = ag.detached_copy()
        ag.altloc, b.altloc = 'A', 'B'
        for a in ag.atoms():
            a.occ = 0.6
        for a in b.atoms():
            a.occ = 0.4
            if a.name.strip() not in ('N', 'CA', 'C', 'O'):
                x, y, z = a.xyz
                a.xyz = (x + 0.4, y - 0.3, z + 0.2)
        rg.append_atom_group(b)
noh = os.path.join(out, 'noh.pdb')
open(noh, 'w').write(h.as_pdb_string())
withh = tools.add_hydrogens(tools.find_tools(), noh, out)
hh = iotbx.pdb.input(file_name=withh).construct_hierarchy()
open(os.path.join(out, 'h.pdb'), 'w').write(hh.as_pdb_string())
"""

FULL = """
import json, os, pickle, sys, tempfile
import numpy as np
import iotbx.pdb
from remolprobity import geometry, io, naming, tools, worker
path = sys.argv[1]
m = io.read_model(path)
d = tempfile.mkdtemp()
block = io.write_blocks(m, [np.arange(len(m.residues))], d)[0]
shared = os.path.join(d, 'shared.pkl')
with open(shared, 'wb') as f:
    pickle.dump({'owner': {k: 0 for k in m.residues}, 'conversion': naming.conversion(m.chain_ids, m.residue_names),
                 'occupancy_max': m.occupancy_max, 'tools': tools.find_tools(),
                 'definitions': ['v34'], 'margin': 15.0}, f)
r = worker.run_block({'block': 0, 'path': block, 'workdir': d, 'shared': shared})
ref = geometry.score('v34', iotbx.pdb.input(file_name=block).construct_hierarchy(), lambda c, i: True)
print(repr({'v34': r['geometry']['v34'], 'ref': ref, 'rota': r['rotamers'], 'rama': r['ramachandran'],
            'probe_atoms': r['probe_atoms']}))
"""


def _run(path):
    return ast.literal_eval(in_image('-c', FULL, str(path)).stdout.strip().splitlines()[-1])


@pytest.fixture(scope='module')
def variants(tmp_path_factory):
    d = tmp_path_factory.mktemp('variants')
    in_image('-c', PREP, os.path.join(DATA, '1crn.pdb'), str(d))
    lines = (d / 'h.pdb').read_text().splitlines(True)
    # the element column, 77-78, blanked: hydrogens only by their names
    blank = [l[:76] + '  ' + l[78:] if l.startswith(('ATOM', 'HETATM')) else l for l in lines]
    (d / 'blank.pdb').write_text(''.join(blank))
    # ARG 10's side-chain end at the end of the file (before TER or END)
    src = open(os.path.join(DATA, '1crn.pdb')).read().splitlines(True)
    late = [l for l in src if l.startswith('ATOM') and l[22:26].strip() == '10'
            and l[12:16].strip() in ('NE', 'CZ', 'NH1', 'NH2')]
    assert len(late) == 4
    rest = [l for l in src if l not in late]
    end = next(i for i, l in enumerate(rest) if l.startswith(('TER', 'END')))
    (d / 'late.pdb').write_text(''.join(rest[:end] + late + rest[end:]))
    return d


@pytest.mark.container
def test_hydrogen_free_copy_does_not_flip_the_block(variants):
    # flipping the hydrogen-free copy must not reach the hierarchy geometry scores
    out = _run(variants / 'h.pdb')
    assert out['v34'] == out['ref']
    assert out['ref']['bonds']['outliers']


@pytest.mark.container
def test_hydrogens_found_by_name_when_elements_are_blank(variants):
    with_elements, blank = _run(variants / 'h.pdb'), _run(variants / 'blank.pdb')
    assert blank['rota'] == with_elements['rota'] and blank['rama'] == with_elements['rama']
    assert blank['probe_atoms'] == with_elements['probe_atoms']


@pytest.mark.container
def test_late_atoms_join_their_residue(variants):
    base = _run(os.path.join(DATA, '1crn.pdb'))
    late = _run(variants / 'late.pdb')
    assert any(tuple(r[:2]) == ('A', '10') for r in late['rota'])
    assert late['rota'] == base['rota'] and late['probe_atoms'] == base['probe_atoms'] == 642
