"""Output formats, and the stock and compare subcommands, in the image"""
import json
import os
import re

import pytest

from helpers import DATA, in_image
from remolprobity import compare, result

SMALL = os.path.join(DATA, '1ehz_10nt.cif')
# the versions the log names; the provenance, which what the log adds leaves as it was
VERSIONS = r'MolProbity \S+ \([0-9a-f]{12}\), cctbx \S+, reduce \S+, probe \S+'
PROVENANCE = {'blocks', 'cctbx', 'deviations', 'geometry', 'input', 'options', 'peak_gb', 'probe', 'reduce',
              'reduce_self_check', 'remolprobity', 'tools', 'wall_seconds'}
STOCK_PROVENANCE = {'cctbx', 'clashscore', 'input', 'stock', 'tools', 'wall_seconds'}


def log_lines(stderr):
    return [line.split('] ', 1)[1] for line in stderr.splitlines() if line.startswith('[remolprobity ')]


def remolprobity(*args, check=True):
    return in_image('-m', 'remolprobity', *args, check=check, timeout=3600, memory_gb=4)


# the budget of a run in in_image's 4 GB scope: from what the scope's limit
# leaves, unless the machine has less available
SCOPE_BUDGET = (r'memory budget (\S+) GB: 90% of the (\S+) GB left under the 4\.3 GB \(4\.0 GiB\) memory limit '
                r'of cgroup /\S+\.scope')
MACHINE_BUDGET = (r"memory budget (\S+) GB: 80% of the (\S+) GB the machine has available \(its cgroup's limit "
                  r"leaves \S+ GB\)")


def machine_available_gb():
    with open('/proc/meminfo') as f:
        return next(int(line.split()[1]) * 1024 / 1e9 for line in f if line.startswith('MemAvailable:'))


def check_budget(line):
    """A budget line of a run in a 4 GB scope: 90% of what the scope leaves, or 80% of what the machine has"""
    scope, machine = re.fullmatch(SCOPE_BUDGET, line), re.fullmatch(MACHINE_BUDGET, line)
    assert scope or machine, line
    if machine_available_gb() > 8:  # far more than the scope: the scope sets the budget
        assert scope, line
    budget, room = map(float, (scope or machine).groups())
    assert 0 < room <= 4.3 if scope else 0 < room
    assert abs(budget - (0.9 if scope else 0.8) * room) <= 0.1, line


@pytest.mark.container
def test_pickle_json_and_stdout_hold_the_same_result(tmp_path):
    remolprobity(SMALL, '-o', tmp_path / 'r.pkl', '--geometry', 'v34,v33')
    remolprobity(SMALL, '-o', tmp_path / 'r.json', '--geometry', 'v34,v33')
    r = remolprobity(SMALL, '-o', '-', '--geometry', 'v34,v33')
    # stdout holds the result alone: the log, and whatever the tools print, go to stderr
    (tmp_path / 'stdout.json').write_text(r.stdout)
    assert json.loads(r.stdout)['schema'] == result.SCHEMA and 'done in' in r.stderr
    pkl, js, out = (result.comparable(result.read(str(tmp_path / n))) for n in ('r.pkl', 'r.json', 'stdout.json'))
    assert pkl == js == out
    assert not [n for n in os.listdir(tmp_path) if n.startswith('remolprobity.')]


@pytest.mark.container
def test_log_names_versions_time_and_memory(tmp_path):
    r = remolprobity(SMALL, '-o', tmp_path / 'r.json')
    lines = log_lines(r.stderr)
    assert re.fullmatch(r'remolprobity \S+ with ' + VERSIONS, lines[0]), lines[0]
    check_budget(lines[1])
    assert not [line for line in lines if line.startswith('warning')], lines
    assert re.fullmatch(r'done in \d+ s: .*', lines[-2]), lines[-2]
    m = re.fullmatch(r'memory about (\S+) GB without page cache: main process (\S+) GB; 1 block at a time, '
                     r'each up to (\S+) GB, and up to (\S+) GB more for reduce or probe', lines[-1])
    assert m, lines[-1]
    total, main, worker, child = map(float, m.groups())
    assert main > 0 and worker > 0 and child > 0 and abs(total - main - worker - child) <= 0.015
    text = (tmp_path / 'r.json').read_text()
    assert set(json.loads(text)['provenance']) == PROVENANCE and set(json.loads(text)['provenance']['peak_gb']) == \
        {'main', 'worker'}
    assert 'children' not in text


@pytest.mark.container
def test_memory_given_is_checked_against_the_scope_and_the_plan(tmp_path):
    lines = log_lines(remolprobity(SMALL, '-o', tmp_path / 'r.json', '--memory', '64').stderr)
    assert lines[1] == 'memory budget 64.0 GB (--memory)', lines
    assert re.fullmatch(r'warning: --memory 64\.0 GB is more than the \S+ GB (left under the 4\.3 GB \(4\.0 GiB\) '
                        r'memory limit of cgroup /\S+\.scope|the machine has available)', lines[2]), lines
    # a budget the plan cannot keep to: one worker, and a warning that says what would need less
    lines = log_lines(remolprobity(SMALL, '-o', tmp_path / 'r.json', '--memory', '0.5').stderr)
    assert lines[1] == 'memory budget 0.5 GB (--memory)' and not lines[2].startswith('warning'), lines
    assert [line for line in lines if re.fullmatch(r'\d+ atoms, \d+ residues: 1 blocks .*, 1 workers, about \S+ GB',
                                                   line)], lines
    assert [line for line in lines if re.fullmatch(r'warning: expected peak \S+ GB exceeds the memory budget of 0\.5 '
                                                   r'GB; smaller blocks \(--block-atoms\) need less', line)], lines


@pytest.mark.container
def test_stock_agrees_with_one_block(tmp_path):
    model = os.path.join(DATA, '4hhb.cif')
    r = in_image('-m', 'remolprobity', 'stock', model, '-o', '-', timeout=3600, memory_gb=4)
    (tmp_path / 'stock.json').write_text(r.stdout)
    stock = result.read(str(tmp_path / 'stock.json'))
    assert stock['provenance']['stock'] is True and stock['provenance']['input']['atoms'] == 4779
    assert set(stock['provenance']) == STOCK_PROVENANCE
    lines = log_lines(r.stderr)
    assert re.fullmatch(r'remolprobity \S+ stock, with ' + VERSIONS, lines[0]), lines[0]
    check_budget(lines[1])
    assert re.fullmatch(r'done in \d+ s: .*', lines[-2]), lines[-2]
    m = re.fullmatch(r'memory about (\S+) GB without page cache: stock MolProbity (\S+) GB \(its largest process\), '
                     r'this process (\S+) GB', lines[-1])
    assert m, lines[-1]
    total, child, own = map(float, m.groups())
    assert child > 0 and own > 0 and abs(total - child - own) <= 0.015
    remolprobity(model, '-o', tmp_path / 'one.pkl', '--geometry', 'v34,v33', '--block-atoms', '0')
    c = in_image('-m', 'remolprobity', 'compare', tmp_path / 'stock.json', tmp_path / 'one.pkl', check=False)
    assert c.returncode == 0 and 'same' in c.stdout, c.stdout
    assert compare.differences(result.comparable(stock), result.comparable(result.read(str(tmp_path / 'one.pkl'))),
                               tol=1e-6) == []
    assert not [n for n in os.listdir(tmp_path) if n.startswith('remolprobity-stock.')]


@pytest.mark.container
def test_stock_refusal_is_one_line(tmp_path):
    bad = tmp_path / 'bad.cif'
    bad.write_text(open(SMALL).read().replace(' 1.00 ', ' .    ', 1))
    r = in_image('-m', 'remolprobity', 'stock', bad, '-o', tmp_path / 'ref.json', check=False, timeout=600)
    assert r.returncode == 1 and 'error:' in r.stderr and 'Traceback' not in r.stderr, r.stderr[-2000:]
    assert not os.path.exists(tmp_path / 'ref.json')
