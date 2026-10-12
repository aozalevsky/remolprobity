"""python -m remolprobity: a run, or the compare and stock subcommands"""
import json
import os
import subprocess
import sys

from helpers import REPO
from test_compare import sample


def remolprobity(*args):
    return subprocess.run([sys.executable, '-m', 'remolprobity'] + list(args), capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=REPO))


def test_compare_subcommand(tmp_path):
    a = tmp_path / 'a.json'
    a.write_text(json.dumps(sample()))
    r = remolprobity('compare', str(a), str(a))
    assert r.returncode == 0 and 'same' in r.stdout


def test_compare_reads_stdin(tmp_path):
    a = tmp_path / 'a.json'
    a.write_text(json.dumps(sample()))
    r = subprocess.run([sys.executable, '-m', 'remolprobity', 'compare', '-', str(a)], input=a.read_text(),
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=REPO))
    assert r.returncode == 0 and 'same' in r.stdout


def test_help_names_the_subcommands():
    r = remolprobity('--help')
    assert r.returncode == 0 and 'remolprobity compare' in r.stdout and 'remolprobity stock' in r.stdout


def test_stock_subcommand_is_dispatched():
    # its help needs no cctbx
    r = remolprobity('stock', '--help')
    assert r.returncode == 0 and r.stdout.startswith('usage: remolprobity stock')
