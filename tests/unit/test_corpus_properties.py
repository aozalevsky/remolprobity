import os
import sys

from test_io_cif import HEAD, ROWS
from test_io_pdb import PDB

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from corpus_properties import properties  # noqa: E402


def test_cif_properties(tmp_path):
    p = tmp_path / 'm.cif'
    p.write_text(HEAD + ROWS)
    got = properties(str(p))
    assert got['format'] == 'cif' and got['atoms'] == 6
    assert got['insertion_codes'] == 1 and got['ligands'] == ['XYZW'] and got['hydrogens'] == 1
    assert got['zero_occupancy'] == 2 and got['long_chain_ids'] == 1 and got['max_resseq'] == 10


def test_pdb_properties(tmp_path):
    p = tmp_path / 'm.pdb'
    p.write_text(PDB)
    got = properties(str(p))
    assert got['format'] == 'pdb' and got['max_resseq'] == 10000 and got['insertion_codes'] == 1
