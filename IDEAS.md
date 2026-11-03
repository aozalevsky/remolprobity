# Ideas

Measured possibilities that are not planned. Each says what it would save and
what it would take.

## The GPU clash path reads each block's file four times

With `--gpu`, a worker scores a block's clashes from reduce's output with
hydrogens (`jax_clashes.score_from_withh`), and reads that file four times:

1. `clashes._coordinates`: every atom's coordinates;
2. `clashes._cliques`: reduce's `USER  MOD` records;
3. `load_probe_atoms`, from `probe_clashes_jax`: every atom, with probe's
   properties, in Python, one atom at a time;
4. `count_probe_atoms`: every atom again, to count the probe atoms the block
   owns.

On one 241,548-atom block of 8WQL, under cProfile, `load_probe_atoms` took
8.5 s, the loop in `probe_clashes_jax` that turns its atoms into arrays 3.0 s,
and `count_probe_atoms` 2.5 s. Without the profiler, the worker's preparation
before the GPU call takes 6–7.5 s per block of 8WQL.

This is now most of what `--gpu` spends on clashes besides reduce. With the GPU
server's requests in threads and the batches gathered on the GPU, the GPU's part
is about 1 s per block. A block's clash stage on 8WQL (8 workers, one L40) is
20–25 s with `--gpu` and 20–24 s with the C probe; reduce takes about 9 s in
both.

What it would take:

- **The count from the first parse.** `count_probe_atoms` takes the atoms not
  named `Q*`, of owned residues, not water, not at 9998, with occupancy above
  `ogt`. That could come from `load_probe_atoms`' atoms, if they include every
  atom the count needs; that has to be checked first. It saves one pass, about
  1–2 s per block.
- **Probe's properties per residue and atom name.** Most of what
  `load_probe_atoms` assigns depends only on the residue and atom names, as
  `identify_atom` already does with its cache; the chain ends, histidine
  charges and hydrogens' bonded atoms depend on neighbours. Computing the first
  kind once per name pair, and the arrays with NumPy, would cut most of the
  rest. It is probe's atom typing, so it needs tests that the arrays are
  identical on the corpus.
