"""io reads a PDB atom record's numbers exactly as iotbx does, and refuses what iotbx cannot read"""
import json

import pytest

from helpers import in_image

DRIVER = r'''
import json
import iotbx.pdb
from remolprobity import io

BASE = 'ATOM      1  CA  ALA A   1      11.104   6.134  -6.504  1.00 10.00           C'
VALUES = ['', '.', '?', '1e3', '1E2', '+1.0', '1.', '.5', '-.5', '1_0', 'inf', 'nan', '0x1', '1.0e', 'e1', '1 0',
          '+', '-', '-0', '1e-2', '1.5d2', '+-1', '1e+2', '٣', 'abc', '12.5', ' -3.25',
          # blanks after the first character: zeros to iotbx
          '12  ', '1.0e ', '. ', '1 e2 ', '- 1 ', '1.  5', '2 2 2', '1 .5 ', '? ', '. 5']
out = []
for column, i, j in io.PDB_NUMBERS:
    for v in VALUES:
        if len(v) > j - i:
            continue
        line = BASE[:i] + v.rjust(j - i) + BASE[j:]
        try:
            a = iotbx.pdb.input(source_info=None, lines=[line]).construct_hierarchy().atoms()[0]
            theirs = {'x': a.xyz[0], 'y': a.xyz[1], 'z': a.xyz[2], 'occupancy': a.occ, 'B': a.b}[column]
        except Exception:
            theirs = None
        out.append([column, v, theirs, io.pdb_number(line[i:j])])
print(json.dumps(out))
'''


@pytest.mark.container
def test_pdb_numbers_as_iotbx_reads_them(tmp_path):
    script = tmp_path / 'pdb_number_cases.py'
    script.write_text(DRIVER)
    cases = json.loads(in_image(str(script)).stdout.strip().splitlines()[-1])
    assert [c for c in cases if c[2] != c[3]] == []
    assert {c[2] is None for c in cases} == {True, False}
