#!/bin/sh
# Build probe from a probe source tree (run in the MolProbity image).
# usage: build_probe.sh SRC OUT
set -e
gcc -O2 -w -o "$2" "$1"/*.c -lm
