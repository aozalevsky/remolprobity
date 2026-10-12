"""The 800k-atom subset of 9A1N: one block against blocks, each run in a 16 GB scope"""
import os

import pytest

from helpers import in_image
from remolprobity import compare, result

CORPUS_DIR = os.environ.get('REMOLPROBITY_CORPUS', os.path.expanduser('~/.cache/remolprobity/corpus'))


def run(model, out, *args):
    in_image('-m', 'remolprobity', model, '-o', out, '--geometry', 'v34,v33', *args,
             timeout=12 * 3600, memory_gb=16)
    return result.read(out)


@pytest.mark.heavy
def test_800k_one_block_against_blocks(tmp_path):
    model = os.path.join(CORPUS_DIR, '9a1n_800k.cif')
    if not os.path.exists(model):
        pytest.skip('make it with: scripts/fetch_corpus.py --tier heavy')
    one = run(model, str(tmp_path / 'one.pkl'), '--block-atoms', '0', '--nproc', '1')
    many = run(model, str(tmp_path / 'many.pkl'), '--memory', '16')
    assert many['provenance']['blocks']['count'] > 4
    assert compare.differences(result.comparable(one), result.comparable(many), tol=1e-9) == []
