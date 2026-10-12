#!/bin/sh
# Prepare a MolProbity sandbox (apptainer build --sandbox) for the tests:
# keep its stock reduce at /opt/remolprobity-stock/reduce, patch reduce and
# probe, and put the patched builds in place of the stock ones.
# usage: scripts/dev_image.sh SANDBOX
set -e
S=$(cd "$1" && pwd)
R=$(cd "$(dirname "$0")/.." && pwd)
M="$S/opt/molprobity"
apply() {
  if patch -d "$1" -p1 -R --dry-run -s < "$2" >/dev/null 2>&1; then
    echo "already applied: $2"
  else
    patch -d "$1" -p1 < "$2"
  fi
}
mkdir -p "$S/opt/remolprobity-stock"
[ -f "$S/opt/remolprobity-stock/reduce" ] || cp -p "$M/build/reduce/exe/reduce" "$S/opt/remolprobity-stock/reduce"
apply "$M/modules/reduce" "$R/patches/reduce-clique-order.patch"
apply "$M/modules/probe" "$R/patches/probe-hisNHcheck-quadratic.patch"
B="$R/.build"; mkdir -p "$B"
apptainer exec "$S" sh "$R/scripts/build_reduce.sh" /opt/molprobity/modules/reduce "$B/reduce"
apptainer exec "$S" sh "$R/scripts/build_probe.sh" /opt/molprobity/modules/probe "$B/probe"
cp "$B/reduce" "$M/build/reduce/exe/reduce"
cp "$B/probe" "$M/build/probe/exe/probe"
echo "patched reduce and probe installed in $S"
