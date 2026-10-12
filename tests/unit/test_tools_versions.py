"""The versions the log names: MolProbity's checkout, reduce and probe"""
import subprocess

from remolprobity import tools

CORE_PHP = '<?php\n//    define("MP_VERSION", "4.01a");\ndefine("MP_VERSION", "4.5.2"); // initialize MP version\n'


def checkout(path, core=CORE_PHP):
    (path / 'lib').mkdir(parents=True)
    (path / 'lib' / 'core.php').write_text(core)
    return str(path)


def git(path, *args):
    return subprocess.run(['git', '-C', str(path)] + list(args), check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(path):
    git(path, 'init', '-q')
    git(path, '-c', 'user.name=t', '-c', 'user.email=t@example.org', 'commit', '-q', '--allow-empty', '-m', 'x')
    return git(path, 'rev-parse', 'HEAD')


def test_version_and_commit(tmp_path):
    root = checkout(tmp_path / 'molprobity')
    assert tools.molprobity_version(root) == 'MolProbity 4.5.2'
    head = commit(root)
    assert tools.molprobity_version(root) == 'MolProbity 4.5.2 (%s)' % head[:12]


def test_commit_alone(tmp_path):
    root = checkout(tmp_path / 'molprobity', core='<?php\n')
    head = commit(root)
    assert tools.molprobity_version(root) == 'MolProbity %s' % head[:12]


def test_not_the_commit_of_a_repository_around_it(tmp_path):
    commit(tmp_path)
    assert tools.molprobity_version(checkout(tmp_path / 'molprobity')) == 'MolProbity 4.5.2'


def test_no_checkout(tmp_path):
    assert tools.molprobity_version(str(tmp_path / 'molprobity')) is None
    (tmp_path / 'molprobity').mkdir()
    commit(tmp_path / 'molprobity')
    assert tools.molprobity_version(str(tmp_path / 'molprobity')) is None


def test_tool_versions_read_plainly():
    assert tools.tool_version('reduce', 'reduce.4.16.250520') == 'reduce 4.16.250520'
    assert tools.tool_version('probe', 'probe.2.26.021123') == 'probe 2.26.021123'
    assert tools.tool_version('probe', 'unknown') == 'probe unknown'
    assert tools.tool_version('reduce', '') == 'reduce unknown'
