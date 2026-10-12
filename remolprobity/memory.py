"""
The memory a run may plan for: what its cgroup's memory limit leaves (cgroup
v2 or v1), and what the machine has available (MemAvailable in
/proc/meminfo). GB is 10^9 bytes, as in the rest of remolprobity; a systemd
MemoryMax=16G is 16 GiB, 17.2 GB.
"""
import os
import re
from collections import namedtuple

CGROUP_SHARE = 0.9  # of what the cgroup's limit leaves
MACHINE_SHARE = 0.8  # of what the machine has available, which other jobs change during the run
NO_LIMIT = 1 << 60  # cgroup v1 reads "no limit" as a number near 2**63

Limit = namedtuple('Limit', 'room limit cgroup')  # bytes left, bytes, the cgroup that sets them
Budget = namedtuple('Budget', 'gb line warning')  # GB or None, for the log, a warning or None


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def _number(path):
    """A cgroup file's number; None if missing, 'max' or not a number"""
    text = _read(path)
    try:
        return int(text.split()[0])
    except (AttributeError, IndexError, ValueError):
        return None


def _stat(path, key):
    """A value of a cgroup's memory.stat; 0 if missing"""
    for line in (_read(path) or '').splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == key and fields[1].isdigit():
            return int(fields[1])
    return 0


def _unescape(field):
    """A path as mountinfo writes it, with a space as \\040"""
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), field)


def _hierarchies(root):
    """
    The cgroup hierarchies that can limit memory, with the process's cgroup
    in each: (version, mount root, mount point, cgroup path). Version 2 is
    the unified hierarchy; version 1 the memory controller's.
    """
    paths = {}
    for line in (_read(os.path.join(root, 'proc/self/cgroup')) or '').splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        if parts[0] == '0' and parts[1] == '':
            paths[2] = parts[2]
        elif 'memory' in parts[1].split(','):
            paths[1] = parts[2]
    for line in (_read(os.path.join(root, 'proc/self/mountinfo')) or '').splitlines():
        fields = line.split()
        if '-' not in fields[6:]:
            continue
        after = fields[fields.index('-', 6) + 1:]
        if not after:
            continue
        if after[0] == 'cgroup2':
            version = 2
        elif after[0] == 'cgroup' and len(after) > 2 and 'memory' in after[2].split(','):
            version = 1
        else:
            continue
        if version in paths:
            yield version, _unescape(fields[3]), _unescape(fields[4]), paths[version]


def _levels(root, mount_root, mount_point, path):
    """The process's cgroup and its parents up to the mount point: (directory, cgroup path)"""
    if '..' in path.split('/'):
        return  # outside what this mount shows
    if mount_root != '/':
        if path != mount_root and not path.startswith(mount_root.rstrip('/') + '/'):
            return
        path = path[len(mount_root):]
    top = os.path.normpath(os.path.join(root, mount_point.lstrip('/')))
    rel = path.strip('/')
    while True:
        yield os.path.join(top, rel) if rel else top, '/' + '/'.join(p for p in (mount_root.strip('/'), rel) if p)
        if not rel:
            return
        rel = os.path.dirname(rel)


def cgroup_limit(root='/'):
    """
    The tightest memory limit on this process's cgroups, cgroup v2 or v1:
    the Limit whose room (limit less usage, inactive file cache not counted
    as usage) is the smallest. None if no cgroup has a limit, or none can be
    read. v2 counts memory.high as a limit too: above it the kernel
    throttles the cgroup.
    """
    tightest = None
    for version, mount_root, mount_point, path in _hierarchies(root):
        for d, name in _levels(root, mount_root, mount_point, path):
            if version == 2:
                limits = [n for n in (_number(os.path.join(d, 'memory.max')),
                                      _number(os.path.join(d, 'memory.high'))) if n is not None]
                usage = _number(os.path.join(d, 'memory.current'))
                inactive = _stat(os.path.join(d, 'memory.stat'), 'inactive_file')
            else:
                limits = [n for n in (_number(os.path.join(d, 'memory.limit_in_bytes')),) if n is not None and
                          n < NO_LIMIT]
                usage = _number(os.path.join(d, 'memory.usage_in_bytes'))
                inactive = _stat(os.path.join(d, 'memory.stat'), 'total_inactive_file')
            if not limits or usage is None:
                continue
            limit = min(limits)
            room = max(0, limit - max(0, usage - inactive))
            if tightest is None or room < tightest.room:
                tightest = Limit(room, limit, name)
    return tightest


def machine_available(root='/'):
    """MemAvailable of /proc/meminfo (bytes); None if not there"""
    for line in (_read(os.path.join(root, 'proc/meminfo')) or '').splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == 'MemAvailable:' and fields[1].isdigit():
            return int(fields[1]) * 1024
    return None


def budget(memory_gb=None, root='/'):
    """
    The memory a run may plan for (GB), with a line for the log and a
    warning or None. memory_gb (--memory) if given, with a warning if it is
    more than the cgroup's limit leaves or the machine has; else the smaller
    of 90% of what the cgroup's limit leaves and 80% of what the machine has
    available; None if neither is known.
    """
    limit, available = cgroup_limit(root), machine_available(root)
    rooms = []  # (GB, share, what it is)
    if limit is not None:
        rooms.append((limit.room / 1e9, CGROUP_SHARE, 'left under the %.1f GB (%.1f GiB) memory limit of cgroup %s'
                      % (limit.limit / 1e9, limit.limit / (1 << 30), limit.cgroup)))
    if available is not None:
        rooms.append((available / 1e9, MACHINE_SHARE, 'the machine has available%s' % (
            '' if memory_gb is not None else ' (no memory limit on its cgroup)' if limit is None else
            " (its cgroup's limit leaves %.1f GB)" % (limit.room / 1e9))))
    if memory_gb is not None:
        tight = min(rooms, default=None)
        warning = None
        if tight is not None and memory_gb > tight[0]:
            warning = 'warning: --memory %.1f GB is more than the %.1f GB %s' % (memory_gb, tight[0], tight[2])
        return Budget(memory_gb, 'memory budget %.1f GB (--memory)' % memory_gb, warning)
    if not rooms:
        return Budget(None, 'memory budget unknown: neither a cgroup memory limit nor /proc/meminfo found', None)
    gb, share, what = min(rooms, key=lambda r: r[0] * r[1])
    return Budget(gb * share, 'memory budget %.1f GB: %d%% of the %.1f GB %s' % (gb * share, round(share * 100), gb,
                                                                             what), None)
