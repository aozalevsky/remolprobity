"""Paths and running code in the MolProbity image, for the tests"""
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'tests', 'data')
IMAGE = os.environ.get('REMOLPROBITY_IMAGE')
STOCK_REDUCE = os.environ.get('REMOLPROBITY_STOCK_REDUCE', '/opt/remolprobity-stock/reduce')


def in_image(*args, check=True, env=None, timeout=None, memory_gb=None):
    """
    mmtbx.python with args in the MolProbity image, remolprobity importable;
    with memory_gb, in a scope the kernel kills beyond that much memory
    """
    cmd = []
    if memory_gb:
        cmd += ['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemoryMax=%dG' % memory_gb,
                '-p', 'MemorySwapMax=0']
    cmd += ['apptainer', 'exec', '--env', 'PYTHONPATH=%s' % REPO]
    for key, value in (env or {}).items():
        cmd += ['--env', '%s=%s' % (key, value)]
    cmd += [IMAGE, 'mmtbx.python'] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise AssertionError('%s failed (%d):\n%s\n%s' % (
            ' '.join(str(a) for a in args), r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
    return r
