"""What the log says at the end of a run: the time it took and its peak memory"""
from remolprobity import run


def test_duration_reads_plainly():
    assert run.duration(0.4) == '0 s'
    assert run.duration(86.4) == '86 s'
    assert run.duration(119.6) == '2 min 00 s'
    assert run.duration(3599) == '59 min 59 s'
    assert run.duration(7200) == '2 h 00 min'
    assert run.duration(7290) == '2 h 01 min'
    assert run.duration(30 * 3600 + 59) == '30 h 00 min'


def test_memory_of_one_block():
    assert run.memory_summary(0.14, [(1.39, 0.29)], 1) == \
        ('memory about 1.82 GB without page cache: main process 0.14 GB; 1 block at a time, each up to 1.39 GB, '
         'and up to 0.29 GB more for reduce or probe')


def test_memory_estimate_takes_the_blocks_that_need_the_most():
    # with their reduce or probe, the blocks need 1.2, 2.1, 2.0 and 0.3 GB; two run at once
    line = run.memory_summary(0.5, [(1.0, 0.2), (2.0, 0.1), (1.5, 0.5), (0.3, 0.0)], 2)
    assert line == ('memory about 4.60 GB without page cache: main process 0.50 GB; 2 blocks at a time, '
                    'each up to 2.00 GB, and up to 0.50 GB more for reduce or probe')


def test_memory_with_more_workers_than_blocks():
    assert run.memory_summary(0.5, [(1.0, 0.25)], 4).startswith(
        'memory about 1.75 GB without page cache: main process 0.50 GB; 1 block at a time')
