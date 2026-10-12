"""Tasks for test_schedule (imported by name in forkserver children)"""
import os
import time


def double(task):
    return task['block'] * 2


def fail(task):
    if task['block'] == 1:
        raise ValueError('block one is broken')
    return task['block']


def die(task):
    os._exit(3)


def sleep(task):
    time.sleep(30)


def spawn_child(task):
    """A worker that leaves a long-lived subprocess behind, then hangs"""
    import subprocess
    child = subprocess.Popen(['sleep', '1000'])
    with open(task['pidfile'], 'w') as f:
        f.write(str(child.pid))
    time.sleep(30)


def sigkill(task):
    import signal
    os.kill(os.getpid(), signal.SIGKILL)


def quick(task):
    """A result small enough to sit in the pipe while the worker exits"""
    time.sleep(0.5)
    return task['block'] * 2


class Refused(Exception):
    """A known problem with a block's data"""


def refuse(task):
    raise Refused('the data of block %d is bad' % task['block'])
