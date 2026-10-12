"""IHMValidation's own code reads the caches (needs its images and the 3.4 checkout)"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'unit'))
from test_adapters import sample  # noqa: E402

from remolprobity.adapters import ihmv33, ihmv34  # noqa: E402

IHMV_IMAGE = os.environ.get('REMOLPROBITY_IHMV_IMAGE')
IHMV34 = os.environ.get('REMOLPROBITY_IHMV34')

pytestmark = pytest.mark.skipif(not IHMV_IMAGE, reason='REMOLPROBITY_IHMV_IMAGE is not set')

CHECK33 = """
import pickle, sys
sys.path.insert(0, '/opt/IHMValidation/ihm_validation')
import molprobity
# the summaries need only .data, not a parsed model (checked against a
# real 3.3 run)
mp = molprobity.GetMolprobityInformation.__new__(molprobity.GetMolprobityInformation)
mp.data = pickle.load(open(sys.argv[1], 'rb'))
s = mp.summarize_mp_data()
print(s['bonds']['outliers'], s['angles']['outliers'], s['bonds']['total'])
"""

CHECK34 = """
import pickle, sys
sys.path.insert(0, sys.argv[2])
from cache import unwrap_cache
import molprobity
data, meta = unwrap_cache(pickle.load(open(sys.argv[1], 'rb')))
assert molprobity.has_residues(data)
g = molprobity.ModelGeometry.from_cache(1, data[1])
print(g.bonds_total, len(g.bond_outliers), len(g.residues))
"""


def run(script, *args):
    r = subprocess.run(['apptainer', 'exec', IHMV_IMAGE, 'python3', '-c', script] + list(args),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout.split()


def test_ihmv33_reads_the_cache(tmp_path):
    p = tmp_path / 'c.pkl'
    p.write_bytes(__import__('pickle').dumps(ihmv33.cache(sample(), 1)))
    assert run(CHECK33, str(p)) == ['1', '1', '7']


@pytest.mark.skipif(not IHMV34, reason='REMOLPROBITY_IHMV34 is not set')
def test_ihmv34_reads_the_cache(tmp_path):
    p = tmp_path / 'c.pkl'
    p.write_bytes(__import__('pickle').dumps(ihmv34.cache(sample(), 1, {'MolProbity': '4.5.2'})))
    assert run(CHECK34, str(p), IHMV34) == ['6', '1', '2']
