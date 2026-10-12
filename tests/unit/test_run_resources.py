"""Workers for a memory budget, from the blocks as planned"""
from remolprobity import run


def test_memory_chooses_workers():
    sizes = [180_000] * 60
    nproc, expected = run.choose_workers(sizes, 8_000_000, 1_000_000, memory_gb=16, cpus=48)
    assert 1 <= nproc < 48 and expected <= 16
    # one more would not fit
    assert run.expected_gb(sizes, nproc + 1, 8_000_000, 1_000_000) > 16


def test_workers_from_the_largest_blocks():
    # a dense model: 130,000-atom cores with a 25 Å margin hold about 2.4 times as many atoms
    sizes = [312_000] * 20
    nproc, expected = run.choose_workers(sizes, 2_600_000, 300_000, memory_gb=16, cpus=48)
    assert nproc == 3 and expected <= 16
    # the largest blocks set the budget, not the typical one
    nproc, expected = run.choose_workers([400_000] * 2 + [50_000] * 30, 2_600_000, 300_000, memory_gb=16, cpus=48)
    assert expected == run.expected_gb([400_000, 400_000] + [50_000] * (nproc - 2), nproc, 2_600_000, 300_000)
    assert expected <= 16 < run.expected_gb([400_000] * 2 + [50_000] * 30, nproc + 1, 2_600_000, 300_000)


def test_the_model_s_residues_cost_every_worker():
    # each worker loads the whole model's ownership and stretch maps
    assert run.worker_gb(100_000, 1_000_000) - run.worker_gb(100_000) == 1_000_000 * run.WORKER_GB_PER_MODEL_RESIDUE


def test_explicit_values_win_and_cpus_and_blocks_cap():
    assert run.choose_workers([1000] * 3, 3000, 300, nproc=8)[0] == 3
    assert run.choose_workers([1000] * 10, 10000, 1000, nproc=4, memory_gb=0.1)[0] == 4
    assert run.choose_workers([1000] * 10, 10000, 1000, cpus=2)[0] == 2
    assert run.choose_workers([1000] * 10, 10000, 1000, memory_gb=100, cpus=4)[0] == 4


def test_one_worker_even_over_budget():
    nproc, expected = run.choose_workers([1_000_000], 1_000_000, 100_000, memory_gb=1, cpus=8)
    assert nproc == 1 and expected > 1


def test_small_model_small_worker():
    _, expected = run.choose_workers([1000], 1000, 100, nproc=1)
    assert expected < 1.5


def test_memory_of_zero_or_less_is_refused(tmp_path, capsys):
    for value in ('0', '-4', 'nan'):
        assert run.main([str(tmp_path / 'missing.cif'), '-o', str(tmp_path / 'out.pkl'), '--memory', value]) == 1
        err = capsys.readouterr().err
        assert 'error: --memory must be more than 0 GB, not %s' % value in err and 'reading' not in err
