#!/usr/bin/env python3
"""
Time and peak memory of remolprobity on a model, in a memory-limited scope.
usage: perf.py MODEL --memory GB [--image IMAGE] [--out DIR]
Prints one JSON line: atoms, wall time, the scope's peak memory, blocks,
workers, margin overhead and the blocks run again with a larger margin
({block: margin}).
"""
import argparse
import json
import os
import pickle
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# run in the scope, then print the scope's peak memory (its cgroup is the shell's own)
SCRIPT = ('apptainer exec --env PYTHONPATH="$1" "$2" mmtbx.python -m remolprobity "$3" -o "$4" --memory "$5" || exit $?; '
          'cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.peak"')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('--memory', type=float, required=True)
    p.add_argument('--image', default=os.environ.get('REMOLPROBITY_IMAGE'))
    p.add_argument('--out', default='.')
    a = p.parse_args()
    if not a.image:
        p.error('give --image or set REMOLPROBITY_IMAGE')
    out = os.path.join(os.path.abspath(a.out), os.path.basename(a.model) + '.result.pkl')
    t = time.time()
    r = subprocess.run(['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemoryMax=%dG' % int(a.memory),
                        '-p', 'MemorySwapMax=0', 'sh', '-c', SCRIPT, 'perf', REPO, a.image,
                        os.path.abspath(a.model), out, '%g' % a.memory],
                       stdout=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise SystemExit('remolprobity failed (%d)' % r.returncode)
    wall = time.time() - t
    with open(out, 'rb') as f:
        prov = pickle.load(f)['provenance']
    print(json.dumps({'model': os.path.basename(a.model), 'atoms': prov['input']['atoms'],
                      'wall_min': round(wall / 60, 1),
                      'peak_gb': round(int(r.stdout.strip().splitlines()[-1]) / 1e9, 1),
                      'blocks': prov['blocks']['count'], 'workers': prov['blocks']['workers'],
                      'overhead': round(prov['blocks']['overhead'], 2),
                      'retried': prov['blocks']['retried']}))


if __name__ == '__main__':
    main()
