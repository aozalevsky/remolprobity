"""The memory budget: what the cgroup's limit leaves, and what the machine has available"""
import pytest

from remolprobity import memory

GIB = 1 << 30
V2_MOUNT = ('46 44 0:30 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:7 - cgroup2 cgroup2 '
            'rw,seclabel,nsdelegate,memory_recursiveprot\n')
SCOPE = '/user.slice/user-1000.slice/user@1000.service/app.slice/run-p1-i1.scope'
SERVICE = '/user.slice/user-1000.slice/user@1000.service'
V1_NO_LIMIT = 9223372036854771712  # what cgroup v1 reads as no limit


def tree(tmp_path, mountinfo='', cgroup='', files=None, available_kb=None):
    """A fake root: /proc/self/mountinfo, /proc/self/cgroup, /proc/meminfo and cgroup files"""
    (tmp_path / 'proc' / 'self').mkdir(parents=True)
    (tmp_path / 'proc' / 'self' / 'mountinfo').write_text(mountinfo)
    (tmp_path / 'proc' / 'self' / 'cgroup').write_text(cgroup)
    if available_kb is not None:
        (tmp_path / 'proc' / 'meminfo').write_text(
            'MemTotal:       65000000 kB\nMemFree:         1000000 kB\nMemAvailable:   %8d kB\n' % available_kb)
    for path, text in (files or {}).items():
        (tmp_path / path.lstrip('/')).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path.lstrip('/')).write_text(str(text) + '\n')
    return str(tmp_path)


def v2(path, max='max', current=0, inactive_file=0, high='max'):
    """A cgroup v2 directory's memory files"""
    d = '/sys/fs/cgroup' + path.rstrip('/')
    return {d + '/memory.max': max, d + '/memory.high': high, d + '/memory.current': current,
            d + '/memory.stat': 'anon 1\nfile 2\ninactive_file %d\nactive_file 3' % inactive_file}


def v2_tree(tmp_path, levels, available_kb=None, cgroup=SCOPE):
    """A process in cgroup SCOPE; levels: {cgroup path: memory files}, the rest without a limit"""
    files = {}
    path = cgroup
    while path not in ('', '/'):
        files.update(v2(path))
        path = path.rsplit('/', 1)[0]
    for path, values in levels.items():
        files.update(v2(path, **values))
    return tree(tmp_path, V2_MOUNT, '0::%s\n' % cgroup, files, available_kb)


def test_v2_limit_of_the_scope(tmp_path):
    root = v2_tree(tmp_path, {SCOPE: {'max': 16 * GIB, 'current': 1_000_000_000, 'inactive_file': 400_000_000}})
    # inactive file cache is reclaimed before the limit is reached
    assert memory.cgroup_limit(root) == memory.Limit(16 * GIB - 600_000_000, 16 * GIB, SCOPE)


def test_v2_a_parent_tighter_than_the_scope(tmp_path):
    root = v2_tree(tmp_path, {SCOPE: {'max': 16 * GIB, 'current': 1_000_000_000},
                              SERVICE: {'max': 8 * GIB, 'current': 6_000_000_000}})
    assert memory.cgroup_limit(root) == memory.Limit(8 * GIB - 6_000_000_000, 8 * GIB, SERVICE)


def test_v2_the_tightest_room_not_the_lowest_limit(tmp_path):
    # the scope's limit is lower, but its parent has less left under its own
    root = v2_tree(tmp_path, {SCOPE: {'max': 4 * GIB, 'current': 1_000_000_000},
                              SERVICE: {'max': 8 * GIB, 'current': 7_000_000_000}})
    assert memory.cgroup_limit(root) == memory.Limit(8 * GIB - 7_000_000_000, 8 * GIB, SERVICE)


def test_v2_no_limit_anywhere(tmp_path):
    assert memory.cgroup_limit(v2_tree(tmp_path, {})) is None


def test_v2_memory_high_below_max(tmp_path):
    # above memory.high the kernel throttles and reclaims hard: as good as a limit
    root = v2_tree(tmp_path, {SCOPE: {'max': 16 * GIB, 'high': 12 * GIB, 'current': 0}})
    assert memory.cgroup_limit(root) == memory.Limit(12 * GIB, 12 * GIB, SCOPE)


def test_v2_usage_above_the_limit_leaves_nothing(tmp_path):
    root = v2_tree(tmp_path, {SCOPE: {'max': GIB, 'current': 2 * GIB}})
    assert memory.cgroup_limit(root) == memory.Limit(0, GIB, SCOPE)


def test_v2_in_a_cgroup_namespace(tmp_path):
    # a container with its own cgroup namespace: its cgroup is / and its limit is at the mount point
    root = tree(tmp_path, V2_MOUNT, '0::/\n', v2('/', max=4 * GIB, current=GIB))
    assert memory.cgroup_limit(root) == memory.Limit(3 * GIB, 4 * GIB, '/')


def test_v2_the_host_root_has_no_memory_files(tmp_path):
    root = tree(tmp_path, V2_MOUNT, '0::/\n', {'/sys/fs/cgroup/cgroup.procs': '1'})
    assert memory.cgroup_limit(root) is None


def test_v1_memory_controller(tmp_path):
    mountinfo = ('30 25 0:26 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:13 - cgroup cgroup '
                 'rw,memory\n'
                 '31 25 0:27 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime shared:14 - cgroup cgroup '
                 'rw,cpu,cpuacct\n')
    cgroup = '12:memory:/slurm/uid_1000/job_5/step_0\n11:cpu,cpuacct:/slurm/uid_1000/job_5/step_0\n'
    d = '/sys/fs/cgroup/memory/slurm'
    files = {d + '/memory.limit_in_bytes': V1_NO_LIMIT, d + '/memory.usage_in_bytes': 30 * GIB,
             d + '/uid_1000/memory.limit_in_bytes': V1_NO_LIMIT, d + '/uid_1000/memory.usage_in_bytes': 30 * GIB,
             d + '/uid_1000/job_5/memory.limit_in_bytes': 8 * GIB,
             d + '/uid_1000/job_5/memory.usage_in_bytes': 3 * GIB,
             d + '/uid_1000/job_5/memory.stat': 'cache 2\ntotal_inactive_file %d\n' % GIB,
             d + '/uid_1000/job_5/step_0/memory.limit_in_bytes': V1_NO_LIMIT,
             d + '/uid_1000/job_5/step_0/memory.usage_in_bytes': 2 * GIB}
    root = tree(tmp_path, mountinfo, cgroup, files)
    assert memory.cgroup_limit(root) == memory.Limit(6 * GIB, 8 * GIB, '/slurm/uid_1000/job_5')


def test_v1_a_mount_of_the_process_s_own_cgroup(tmp_path):
    # a container without a cgroup namespace: the mount's root is the process's cgroup
    mountinfo = ('700 690 0:26 /docker/abc /sys/fs/cgroup/memory ro,nosuid,nodev,noexec,relatime master:13 - cgroup '
                 'cgroup rw,memory\n')
    files = {'/sys/fs/cgroup/memory/memory.limit_in_bytes': 2 * GIB,
             '/sys/fs/cgroup/memory/memory.usage_in_bytes': GIB}
    root = tree(tmp_path, mountinfo, '9:memory:/docker/abc\n', files)
    assert memory.cgroup_limit(root) == memory.Limit(GIB, 2 * GIB, '/docker/abc')


def test_hybrid_v1_memory_and_v2_without_it(tmp_path):
    mountinfo = ('29 25 0:25 / /sys/fs/cgroup/unified rw,nosuid,nodev,noexec,relatime shared:5 - cgroup2 cgroup2 rw\n'
                 '30 25 0:26 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:13 - cgroup cgroup '
                 'rw,memory\n')
    files = {'/sys/fs/cgroup/unified/job/cgroup.procs': '1',
             '/sys/fs/cgroup/memory/job/memory.limit_in_bytes': 4 * GIB,
             '/sys/fs/cgroup/memory/job/memory.usage_in_bytes': 0}
    root = tree(tmp_path, mountinfo, '9:memory:/job\n0::/job\n', files)
    assert memory.cgroup_limit(root) == memory.Limit(4 * GIB, 4 * GIB, '/job')


def test_a_mount_point_with_a_space(tmp_path):
    mountinfo = V2_MOUNT.replace(' /sys/fs/cgroup ', r' /sys/fs/c\040g ')
    root = tree(tmp_path, mountinfo, '0::/a\n', {'/sys/fs/c g/a/memory.max': GIB, '/sys/fs/c g/a/memory.current': 0})
    assert memory.cgroup_limit(root) == memory.Limit(GIB, GIB, '/a')


def test_a_cgroup_outside_the_mount_is_not_read(tmp_path):
    # the process's cgroup is above what this namespace mounted
    root = tree(tmp_path, V2_MOUNT, '0::/../other\n', v2('/', max=GIB))
    assert memory.cgroup_limit(root) is None


def test_nothing_to_read(tmp_path):
    root = str(tmp_path)
    assert memory.cgroup_limit(root) is None and memory.machine_available(root) is None
    assert memory.budget(root=root) == memory.Budget(
        None, 'memory budget unknown: neither a cgroup memory limit nor /proc/meminfo found', None)


def test_machine_available(tmp_path):
    assert memory.machine_available(tree(tmp_path, available_kb=50_000_000)) == 50_000_000 * 1024


def test_budget_from_the_cgroup(tmp_path):
    root = v2_tree(tmp_path, {SCOPE: {'max': 16 * GIB, 'current': 180_000_000}}, available_kb=50_000_000)
    b = memory.budget(root=root)
    assert b.gb == pytest.approx(0.9 * (16 * GIB - 180_000_000) / 1e9) and b.warning is None
    assert b.line == ('memory budget 15.3 GB: 90%% of the 17.0 GB left under the 17.2 GB (16.0 GiB) memory limit '
                      'of cgroup %s' % SCOPE)


def test_budget_from_the_machine_under_a_looser_limit(tmp_path):
    root = v2_tree(tmp_path, {SCOPE: {'max': 32 * GIB, 'current': 0}}, available_kb=15_000_000)
    b = memory.budget(root=root)
    assert b.gb == pytest.approx(0.8 * 15_000_000 * 1024 / 1e9)
    assert b.line == ("memory budget 12.3 GB: 80% of the 15.4 GB the machine has available (its cgroup's limit "
                      "leaves 34.4 GB)")


def test_budget_from_the_machine_without_a_limit(tmp_path):
    b = memory.budget(root=v2_tree(tmp_path, {}, available_kb=50_000_000))
    assert b.gb == pytest.approx(0.8 * 50_000_000 * 1024 / 1e9)
    assert b.line == ('memory budget 41.0 GB: 80% of the 51.2 GB the machine has available (no memory limit on '
                      'its cgroup)')


def test_budget_from_the_cgroup_alone(tmp_path):
    b = memory.budget(root=v2_tree(tmp_path, {SCOPE: {'max': 4 * GIB, 'current': 0}}))
    assert b.gb == pytest.approx(0.9 * 4 * GIB / 1e9)


def test_memory_given_is_the_budget(tmp_path):
    root = v2_tree(tmp_path, {SCOPE: {'max': 16 * GIB, 'current': 180_000_000}}, available_kb=50_000_000)
    assert memory.budget(12, root=root) == memory.Budget(12, 'memory budget 12.0 GB (--memory)', None)
    # more than 90% of what is left is the caller's choice; more than all of it gets a warning
    assert memory.budget(16.5, root=root).warning is None
    assert memory.budget(40, root=root).warning == (
        'warning: --memory 40.0 GB is more than the 17.0 GB left under the 17.2 GB (16.0 GiB) memory limit of '
        'cgroup %s' % SCOPE)


def test_memory_given_more_than_the_machine_has(tmp_path):
    b = memory.budget(60, root=v2_tree(tmp_path, {}, available_kb=50_000_000))
    assert b.gb == 60 and b.warning == 'warning: --memory 60.0 GB is more than the 51.2 GB the machine has available'


def test_memory_given_with_nothing_to_check_it_against(tmp_path):
    assert memory.budget(8, root=str(tmp_path)) == memory.Budget(8, 'memory budget 8.0 GB (--memory)', None)
