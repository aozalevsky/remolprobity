"""
Running blocks: one process each, at most nproc at once. Processes come
from a forkserver: forked from a process with threads, a child can inherit
a lock another thread holds at that moment (stderr's) and hang on it for
good. Every exit is checked, so a worker killed for its memory is an error
naming its block, not a hang.

Each worker leads its own process group, so killing a worker also kills
the reduce and probe processes it started; none is left behind as an orphan.
"""
import multiprocessing
import multiprocessing.connection
import os
import pickle
import signal
import time
import traceback


class BlockError(Exception):
    """A block that failed: its error, a crash, or a timeout"""


def _child(fn, task, conn, known=()):
    os.setpgrp()
    try:
        conn.send(('ok', fn(task)))
    except known as e:  # a known problem with the data: its message, the traceback apart
        conn.send(('refused', (str(e), traceback.format_exc())))
    except BaseException:
        conn.send(('error', traceback.format_exc()))
    finally:
        conn.close()


def _path(store, task):
    return os.path.join(store, 'block%05d.pkl' % task['block'])


def _save(path, value):
    temp = '%s.%d.tmp' % (path, os.getpid())
    with open(temp, 'wb') as f:
        pickle.dump(value, f, protocol=4)
    os.replace(temp, path)


def _kill(proc):
    """
    Kill the worker's whole process group, then the worker, and reap it.
    Also after the worker died: its group lives on while a member does,
    and Linux reuses no PID that is a live group's id
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.kill()
    proc.join()


def _reap(proc, grace=10):
    """Wait for a worker that is done; kill it if it lingers past the grace"""
    proc.join(timeout=grace)
    if proc.is_alive():
        _kill(proc)


def _lost(b, proc):
    """The error for a worker that ended without sending a result"""
    hint = ' (killed: out of memory?)' if proc.exitcode == -9 else ''
    return BlockError('block %d ended without a result, exit code %s%s' % (b, proc.exitcode, hint))


def run_tasks(fn, tasks, nproc, timeout=None, store=None, log=None, preload=(), describe=None, known=()):
    """
    fn(task) for every task, in their own processes; results in task order.
    log(message) hears of every block done, with describe(result) and an
    estimate of the time left. An exception of a type in known (a problem
    with the data, not with the code) stops the run with its message alone;
    its traceback goes to the store, as block#####.traceback.txt. Any other
    carries its traceback in the error.
    """
    if nproc < 1:
        raise ValueError('nproc must be at least 1, not %r' % (nproc,))
    numbers = [t['block'] for t in tasks]
    if len(set(numbers)) != len(numbers):
        raise ValueError('duplicate block numbers in tasks')
    ctx = multiprocessing.get_context('forkserver')
    if preload:
        ctx.set_forkserver_preload(list(preload))
    log = log or (lambda message: None)
    results, pending = {}, []
    if store:
        os.makedirs(store, exist_ok=True)
    for t in tasks:
        if store and os.path.exists(_path(store, t)):
            try:
                with open(_path(store, t), 'rb') as f:
                    results[t['block']] = pickle.load(f)
            except Exception as e:
                raise BlockError(
                    'stored result of block %d (%s) is unreadable: %s; '
                    'delete it to run the block again' % (t['block'], _path(store, t), e))
        else:
            pending.append(t)
    if results:
        log('%d of %d blocks done before' % (len(results), len(tasks)))
    running = {}
    start_all, ran = time.time(), 0
    try:
        while pending or running:
            while pending and len(running) < nproc:
                t = pending.pop(0)
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_child, args=(fn, t, send, tuple(known)), name='block%d' % t['block'])
                proc.start()
                send.close()
                running[t['block']] = (proc, recv, t, time.time())
            multiprocessing.connection.wait(
                [r for _, r, _, _ in running.values()] + [p.sentinel for p, _, _, _ in running.values()],
                timeout=1.0)
            for b, (proc, recv, t, start) in list(running.items()):
                # Liveness first: a worker found dead here has written all it
                # ever will, and the pipe keeps it, so poll() below sees its
                # result (or EOF). Asked after poll(), a worker that sent a
                # small result and exited in between would look lost.
                alive = proc.is_alive()
                if recv.poll():
                    try:
                        status, value = recv.recv()
                    except (EOFError, OSError):
                        # the worker died while sending: a partial message, or none
                        status, value = None, None
                    del running[b]
                    if status is None:
                        _kill(proc)  # its reduce or probe may outlive it
                        raise _lost(b, proc)
                    _reap(proc)
                    if status == 'error':
                        raise BlockError('block %d failed:\n%s' % (b, value))
                    if status == 'refused':
                        message, trace = value
                        if store:
                            path = os.path.join(store, 'block%05d.traceback.txt' % b)
                            with open(path, 'w') as f:
                                f.write(trace)
                            message += ' (traceback in %s)' % path
                        raise BlockError('block %d: %s' % (b, message))
                    results[b] = value
                    if store:
                        _save(_path(store, t), value)
                    ran += 1
                    left = (time.time() - start_all) / ran * (len(pending) + len(running))
                    log('block %d done (%d/%d)%s; about %.0f min left'
                        % (b, len(results), len(tasks), describe(value) if describe else '', left / 60))
                elif not alive:
                    del running[b]
                    _kill(proc)  # its reduce or probe may outlive it
                    raise _lost(b, proc)
                elif timeout and time.time() - start > timeout:
                    del running[b]
                    _kill(proc)
                    raise BlockError('block %d timed out after %.0f s' % (b, timeout))
    finally:
        for proc, recv, _, _ in running.values():
            _kill(proc)
    return [results[t['block']] for t in tasks]
