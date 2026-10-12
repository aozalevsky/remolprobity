#!/bin/sh
# Build reduce from a reduce source tree (run in the MolProbity image).
# usage: build_reduce.sh SRC OUT
set -e
SRC=$1; OUT=$2
T=$(mktemp -d)
D="-DAROMATICS_ACCEPT_HBONDS -DCHARFUNCMACROS -DBOOLPREDEFINED -DBRACKETOPERPARMS -DLEFT_JUSTIFY_NUC_RES_OK -DLISTFRIENDFIX -DINCTEMPLATEDEFNS"
I="-I$SRC/toolclasses -I$SRC/libpdb -I$SRC/reduce_src"
gcc -O2 -w -c "$SRC/reduce_src/hybrid_36_c.c" -o "$T/hybrid_36_c.o"
for f in GraphToHoldScores reduce CTab ElementInfo StdResH ResBlk AtomConn AtomPositions DotSph Mover Rot Rot3Fold RotDonor FlipMemo CliqueList AtomDescr PDBrec MoveableNode RotAromMethyl main; do
  g++ -O2 -w -std=c++11 $D $I -c "$SRC/reduce_src/$f.cpp" -o "$T/$f.o"
done
for f in "$SRC"/toolclasses/*.cpp "$SRC"/libpdb/*.cpp; do
  g++ -O2 -w -std=c++11 $D $I -c "$f" -o "$T/$(basename "$f" .cpp).o"
done
g++ -o "$OUT" "$T"/*.o -lm
rm -rf "$T"
