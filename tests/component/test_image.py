import pytest

from helpers import STOCK_REDUCE, in_image


@pytest.mark.container
def test_remolprobity_imports_in_the_image():
    r = in_image('-c', 'import remolprobity, scipy, iotbx.pdb; print(remolprobity.__version__)')
    assert r.stdout.strip() == '0.1.0'


@pytest.mark.container
def test_stock_reduce_is_kept():
    r = in_image('-c', 'import os, sys; sys.exit(0 if os.path.isfile(%r) else 1)' % STOCK_REDUCE, check=False)
    assert r.returncode == 0
