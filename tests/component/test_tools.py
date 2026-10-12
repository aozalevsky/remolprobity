import json
import os

import pytest

from helpers import DATA, STOCK_REDUCE, in_image

CHECK = """
import json, sys, tempfile
from remolprobity import tools
t = tools.find_tools(reduce=sys.argv[1] or None)
print(json.dumps(dict(tools.self_check(t, tempfile.mkdtemp()), version=t.reduce_version)))
"""


@pytest.mark.container
def test_patched_reduce_passes_the_self_check():
    out = json.loads(in_image('-c', CHECK, '').stdout.strip().splitlines()[-1])
    assert out['passed'] and out['near'] == out['far'] == '-28'


@pytest.mark.container
def test_stock_reduce_fails_the_self_check():
    out = json.loads(in_image('-c', CHECK, STOCK_REDUCE).stdout.strip().splitlines()[-1])
    assert not out['passed'] and out['near'] == '-28' and out['far'] == '-36'


@pytest.mark.container
def test_a_missing_or_unrunnable_tool_is_a_clear_error(tmp_path):
    model, out = os.path.join(DATA, '1ehz_10nt.cif'), str(tmp_path / 'r.pkl')
    missing = str(tmp_path / 'nothing')
    r = in_image('-m', 'remolprobity', model, '-o', out, '--reduce', missing, check=False)
    assert r.returncode == 1 and 'error: reduce not found: %s' % missing in r.stderr, r.stderr[-2000:]
    assert 'Traceback' not in r.stderr
    p = tmp_path / 'probe'
    p.write_text('#!/bin/sh\n')
    p.chmod(0o644)
    r = in_image('-m', 'remolprobity', model, '-o', out, '--probe', str(p), check=False)
    assert r.returncode == 1 and 'error: probe is not executable: %s' % p in r.stderr, r.stderr[-2000:]
    assert 'Traceback' not in r.stderr and not os.path.exists(out)
