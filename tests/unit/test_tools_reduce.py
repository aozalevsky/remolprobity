"""run_reduce with stand-ins for reduce: what counts as success"""
import os
import stat

import pytest

from remolprobity import tools

ATOMS = ('ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N\n'
         'ATOM      2  CA  THR A   1      16.967  12.784   4.338  1.00 10.80           C\n'
         'TER\n')


def fake(tmp_path, body):
    """A reduce that runs this shell body on its input (stdin)"""
    p = tmp_path / 'reduce'
    p.write_text('#!/bin/sh\n' + body + '\n')
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return tools.Tools([str(p), '-DB', '/dev/null', '-ALLALT'], 'probe', 'fake', 'fake')


def run(tmp_path, body):
    fin, fout = tmp_path / 'in.pdb', tmp_path / 'out.pdb'
    fin.write_text(ATOMS)
    tools.run_reduce(fake(tmp_path, body), ['-'], str(fin), str(fout))
    return fout.read_text()


def test_a_nonzero_exit_is_success_with_complete_output(tmp_path):
    # reduce exits 255 after -trim and 1 when it gives up on a clique
    out = run(tmp_path, 'cat; exit 255')
    assert out.startswith(ATOMS) and out.rstrip().endswith('END')


def test_killed_by_a_signal_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match='signal 9'):
        run(tmp_path, 'cat; kill -9 $$')


def test_output_cut_short_is_an_error(tmp_path):
    # as a full disk leaves it: reduce's writes fail silently and it exits normally
    with pytest.raises(RuntimeError, match='incomplete'):
        run(tmp_path, 'head -c 100; exit 0')


def test_no_atoms_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match='wrote no atoms'):
        run(tmp_path, 'cat > /dev/null; echo END; exit 100')


def test_an_input_that_ends_with_end_gets_no_second_one(tmp_path):
    fin, fout = tmp_path / 'in.pdb', tmp_path / 'out.pdb'
    fin.write_text(ATOMS + 'END\n')
    tools.run_reduce(fake(tmp_path, 'cat'), ['-'], str(fin), str(fout))
    assert fout.read_text() == ATOMS + 'END\n'


def test_a_missing_or_unrunnable_reduce_is_named(tmp_path):
    with pytest.raises(RuntimeError, match='cannot run reduce'):
        tools._version([str(tmp_path / 'missing'), '-version'], 'reduce')
    p = tmp_path / 'not_executable'
    p.write_text('#!/bin/sh\n')
    with pytest.raises(RuntimeError, match='cannot run reduce'):
        tools._version([str(p), '-version'], 'reduce')
