import os
import time

import pytest

import schedule_tasks as T
from remolprobity import schedule

TASKS = [{'block': b} for b in range(5)]


def test_results_in_task_order():
    assert schedule.run_tasks(T.double, TASKS, 3) == [0, 2, 4, 6, 8]


def test_error_names_the_block_and_carries_the_traceback():
    with pytest.raises(schedule.BlockError, match='block 1 failed') as e:
        schedule.run_tasks(T.fail, TASKS, 2)
    assert 'block one is broken' in str(e.value)


def test_a_dead_worker_is_an_error():
    with pytest.raises(schedule.BlockError, match='exit code 3'):
        schedule.run_tasks(T.die, TASKS[:1], 1)


def test_timeout():
    with pytest.raises(schedule.BlockError, match='timed out'):
        schedule.run_tasks(T.sleep, TASKS[:1], 1, timeout=2)


def test_log_describes_blocks_and_time_left():
    lines = []
    schedule.run_tasks(T.double, TASKS[:2], 1, log=lines.append, describe=lambda r: ': result %d' % r)
    assert any(l.startswith('block 1 done (2/2): result 2') and 'left' in l for l in lines)


def test_store_and_resume(tmp_path):
    store = str(tmp_path / 'results')
    assert schedule.run_tasks(T.double, TASKS, 2, store=store) == [0, 2, 4, 6, 8]
    assert sorted(os.listdir(store)) == ['block%05d.pkl' % b for b in range(5)]
    # everything is stored: nothing runs, so a failing function is never called
    assert schedule.run_tasks(T.die, TASKS, 2, store=store) == [0, 2, 4, 6, 8]


def test_timeout_kills_the_workers_subprocesses(tmp_path):
    pidfile = str(tmp_path / 'pid')
    with pytest.raises(schedule.BlockError, match='timed out'):
        schedule.run_tasks(T.spawn_child, [{'block': 0, 'pidfile': pidfile}], 1, timeout=2)
    with open(pidfile) as f:
        pid = int(f.read())
    deadline = time.time() + 5
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        assert time.time() < deadline, 'subprocess %d survived the timeout' % pid
        time.sleep(0.05)


def test_a_killed_worker_gets_the_out_of_memory_hint():
    with pytest.raises(schedule.BlockError, match='exit code -9') as e:
        schedule.run_tasks(T.sigkill, TASKS[:1], 1)
    assert 'out of memory' in str(e.value)


def test_nproc_must_be_positive():
    with pytest.raises(ValueError):
        schedule.run_tasks(T.double, TASKS, 0)


def test_duplicate_blocks_are_refused():
    with pytest.raises(ValueError):
        schedule.run_tasks(T.double, TASKS + TASKS[:1], 1)


def test_unreadable_stored_result_is_named(tmp_path):
    store = str(tmp_path)
    with open(os.path.join(store, 'block00000.pkl'), 'wb') as f:
        f.write(b'\x80\x04')
    with pytest.raises(schedule.BlockError, match='unreadable'):
        schedule.run_tasks(T.double, TASKS[:1], 1, store=store)


def test_a_worker_that_sent_its_result_and_exited_is_not_lost(monkeypatch):
    # Replays the window between recv.poll() and proc.is_alive(): the first
    # wait times out while the worker still runs, and is_alive() is slow, as
    # if the parent were descheduled; the worker sends a small result and
    # exits in between. Its result is in the pipe and must be taken.
    import multiprocessing.connection
    import multiprocessing.context
    real_wait = multiprocessing.connection.wait
    calls = []

    def wait(objects, timeout=None):
        calls.append(timeout)
        return [] if len(calls) == 1 else real_wait(objects, timeout)

    real_alive = multiprocessing.context.ForkServerProcess.is_alive

    def slow_alive(self):
        time.sleep(2)
        return real_alive(self)

    monkeypatch.setattr(multiprocessing.connection, 'wait', wait)
    monkeypatch.setattr(multiprocessing.context.ForkServerProcess, 'is_alive', slow_alive)
    assert schedule.run_tasks(T.quick, [{'block': 7}], 1) == [14]


def test_a_worker_lost_without_a_result_has_its_group_killed(monkeypatch):
    # the lost path kills the worker's process group, as a timeout does
    killed = []
    real_killpg = os.killpg

    def killpg(pgid, sig):
        killed.append(pgid)
        return real_killpg(pgid, sig)

    monkeypatch.setattr(schedule.os, 'killpg', killpg)
    with pytest.raises(schedule.BlockError, match='exit code 3'):
        schedule.run_tasks(T.die, TASKS[:1], 1)
    assert killed


def test_a_known_refusal_is_its_message_and_the_traceback_is_kept(tmp_path):
    store = str(tmp_path)
    with pytest.raises(schedule.BlockError) as e:
        schedule.run_tasks(T.refuse, TASKS[:1], 1, store=store, known=(T.Refused,))
    path = os.path.join(store, 'block00000.traceback.txt')
    assert str(e.value) == 'block 0: the data of block 0 is bad (traceback in %s)' % path
    with open(path) as f:
        kept = f.read()
    assert 'Traceback' in kept and 'Refused: the data of block 0 is bad' in kept
    # nothing stored as the block's result
    assert not os.path.exists(os.path.join(store, 'block00000.pkl'))


def test_an_unknown_error_keeps_its_traceback_in_the_message():
    with pytest.raises(schedule.BlockError) as e:
        schedule.run_tasks(T.refuse, TASKS[:1], 1)
    assert 'Traceback' in str(e.value) and 'Refused: the data of block 0 is bad' in str(e.value)
