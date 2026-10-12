# remolprobity: MolProbity validation of very large models in spatial blocks

The design of remolprobity, written before it was built (2026-10-10) and
kept up to date with it since. The [README](../README.md) documents its
use; where the two differ, the README describes the code.

## Purpose

MolProbity (`molprobity.molprobity`, cctbx) processes a model whole. Its
memory grows with the model (about 20 KB per atom in the pipeline that
IHMValidation runs) and parts of it are quadratic, so multi-million-atom
integrative models take hours and tens to hundreds of GB, or fail.

remolprobity computes the same validation (Ramachandran, rotamers,
clashes and clashscore, bond and angle deviations) in spatial blocks: each
block is a core plus a margin, processed in its own process, keeping what
its core owns. Memory is bounded by the block size, blocks run in parallel,
and the results are identical to a whole-model run.

It is also a test bed for three proposed changes to cctbx's `ramalyze`
and `clashscore` (see the README's Related work).

### Users

- IHMValidation, which calls it for atomic models and writes its MolProbity
  cache from the result (3.3 format, used by the current images, and 3.4).
- Anyone validating a very large model with MolProbity.

### Success criteria

1. On every model of the test corpus, the result of one block equals stock
   MolProbity's (oracle 1), apart from the documented deviations below.
2. The result does not depend on the block layout: any number of blocks,
   any margin at or above the minimum, gives the same result, every
   identity and count exact and every value within 1e-9 (oracle 2).
   Values can differ in their last bits because cctbx moves each block's
   coordinates into its own P1 box to build restraints (up to 2e-13 Å).
3. 9A1N (1.9M atoms) runs in block mode within 16 GB.
4. A run that cannot guarantee 1 and 2 fails with a clear message rather
   than writing a result.

### Non-goals

- Multi-model input (the caller splits models).
- MolProbity's full report (percentiles, kinemages, Coot scripts).
- Validation against experimental data (maps, structure factors).
- Building or patching reduce and probe (the MolProbity installation
  provides them).

## Decisions taken

| Question | Decision |
|---|---|
| Where it lives | A standalone package, runs under MolProbity's `mmtbx.python`, depends only on cctbx, numpy, scipy |
| Output | A neutral, versioned result; adapters write IHMValidation 3.3 and 3.4 caches |
| Input | Any single-model mmCIF or PDB file cctbx reads; multi-model input is rejected |
| reduce, probe | Taken from the MolProbity installation; reduce must pass a self-check (below), or block mode is refused |
| Engine | Calls the cctbx validators directly per block (not `molprobity.molprobity` per block) |
| Whole model | Is one block, through the same code |

## 1. Package layout

| Module | Responsibility | Depends on |
|---|---|---|
| `io` | Stream mmCIF `atom_site` or PDB `ATOM`/`HETATM` rows; build the atom table (coordinates, residue keys, chain and residue name order) in numpy chunks; reject multi-model input; write a block file for a set of residues | stdlib, numpy (iotbx's polymer residue classes when available) |
| `blocks` | Residue positions (the residue's first CA atom, else the mean of its atoms, as `io` records them); k-d median splits to a target core size; margin residues by distance; ownership map | numpy, scipy |
| `tools` | Locate reduce and probe; reduce self-check; versions | subprocess |
| `worker` | One block: parse; Ramachandran per chain, rotamers; reduce and probe with the global PDB name table; requested geometry definitions; margin checks; owned results as plain data | cctbx |
| `result` | Result schema v1; merge of block results; canonical order; totals; provenance | stdlib |
| `run` | Plan, write blocks, schedule workers, merge, write the result; progress log; resume | the above |
| `adapters.ihmv33`, `adapters.ihmv34` | IHMValidation caches from a result | stdlib |

Command line:

```
mmtbx.python -m remolprobity MODEL -o RESULT [--geometry v34,v33]
    [--memory GB] [--block-atoms N] [--margin A] [--nproc N]
    [--reduce BIN] [--probe BIN] [--workdir DIR] [--keep-workdir] [--resume]
    [--block-timeout S] [--zero-occupancy-as-one] [--missing-b-as-zero]
mmtbx.python -m remolprobity stock MODEL -o RESULT [--workdir DIR]
python3 -m remolprobity compare A B [--tol T] [--limit N] [--json]
python3 -m remolprobity.adapters.ihmv33 RESULT MODEL_ID OUT.mp.pkl
python3 -m remolprobity.adapters.ihmv34 RESULT MODEL_ID OUT.mp.pkl
    [--software KEY=VALUE ...] [--ihmv-version V]
```

The layered image's run script is `mmtbx.python -m remolprobity`.
- **`stock`** writes stock MolProbity's result on the whole model in the
  same schema. It is oracle 1's reference.
- **`compare`** prints both summaries and the differences, and exits with 0
  when the results are the same.

The package grew out of prototype scripts, run on 9A1N subsets of 100k
to 800k atoms.

## 2. Data flow and result

1. `io` streams the model once into the atom table, refusing what iotbx
   (and so stock MolProbity) cannot read: in mmCIF, `.` or `?` in
   coordinates, occupancy or B, or an atom_site item iotbx needs missing;
   in PDB, coordinates, occupancy or B that are not numbers as iotbx reads
   them.
2. `blocks` plans cores and margins.
3. `io` streams the model again, writing each block's residues to its file.
4. Workers process blocks; each returns a block result of what it owns, or
   the margin its core needs when its own is too small (section 3).
5. A block that asked for a larger margin is planned again with it (same
   core) and run again, until every block has its result.
6. `result.merge` combines them in canonical order and checks ownership.
7. The result is written; adapters read it.

Canonical order: residues in file order; clashes by atom pair; outliers by
atoms. Because of it, the result does not depend on the block layout, up
to the last bits of values computed from restraints (success criterion 2).

### Result, schema version 1

Plain Python data (lists, tuples, dicts, numbers, strings), pickled. A
`-o` name ending in `.json`, or `-` (stdout), gives JSON instead. `read`
takes either and restores tuples.

- `schema`: `"remolprobity.result"`, `version`: 1.
- `provenance`: remolprobity version; cctbx commit; reduce and probe
  versions and the self-check result; input path, SHA-256, atom, residue
  and chain counts; block parameters, margin overhead and the margin of
  each block run again with a larger one (`blocks.retried`); geometry
  definitions computed; deviations from stock MolProbity (section 3);
  options that change data (`--zero-occupancy-as-one`,
  `--missing-b-as-zero`); wall time, peak memory.
- `ramachandran`: `total`, `favored`, `allowed`, `outliers`, and every
  result `(chain, resid, resname, altloc, class, phi, psi, score)`, as
  ramalyze returns them, altloc variants included.
- `rotamers`: counts and every result
  `(chain, resid, resname, altloc, evaluation, name, score)`.
- `clashes`: `clashscore`, `atoms` (probe's count, with hydrogens), and
  `(atom1, atom2, overlap, distance)`; an atom is
  `(chain, resid, resname, name, altloc)` in the model's names.
- `geometry`, per definition requested:
  - `v34`: as IHMValidation 3.4 (`mp_validate_bonds` scoring of the
    restraints; conformation-dependent library; symmetric side chains not
    flipped; the restraints mp_validate_bonds scores in the
    installed cctbx (covalent geometry and SS BOND on cctbx from
    2026-08-19); outliers beyond 5 sigma as
    `(atoms..., observed, ideal, sigma, z)`; totals without hydrogens).
  - `v33`: as IHMValidation 3.3 (`restraints.combined`; no
    conformation-dependent library, which `molprobity.molprobity` enables
    only when the file says it was used in refinement; flipped; outliers
    are covalent-geometry restraints beyond 4 sigma, as
    `(atoms..., observed, ideal, |z|)`; totals over covalent-type
    restraints).
- `residues`: per residue in file order, `(chain, resid, resname)` and, for
  `v34`, bonds and angles scored and their sum of Z² (what IHMValidation
  3.4 reports per residue; 3.3 has no per-residue geometry).

Adapters reshape only: 3.4 keeps the first altloc per residue and drops the
altloc field; 3.3 also drops clash distances and takes `v33` geometry.

## 3. Correctness

### Ownership

- A residue is owned by the block whose core holds its position. Merging
  asserts every input residue is owned exactly once.
- Ramachandran and rotamer results belong to their residue; a bond or
  angle to the residue of its first atom in cctbx's order; a clash to its
  first atom in probe's order. Probe's order is the same in every block
  because one table, built from the whole model's chain and residue names,
  gives the PDB names reduce and probe see.

### Margin (default 25 Å), checked per block, enlarged per block

- Reduce cliques: from reduce's `USER  MOD Set` records, a clique with a
  member within contact reach (5 Å) of the core — an owned member, or one
  an owned atom can touch — must have every member farther than contact
  reach from the block's outer edge. Otherwise the block names the clique
  and a margin that would contain it. A clique's members can lie up to
  about 20 Å apart, so a margin of 10 Å plus the clique's extent holds it;
  how much a model needs depends on its hydrogen-bond networks, with no
  bound. 18 Å sufficed for 2,500- to 20,000-atom blocks of the 9A1N L/L10
  excerpt; 4HHB in 200-300-atom blocks needed 21 Å; 1IGT and 100k- to
  400k-atom subsets of 9A1N needed 21-23 Å at 5,000 to 130,000 atoms per
  block, and 25 Å held every layout tried.
- A block whose margin is too small is planned again with the margin it
  names (the same core, so the same residues owned; members at that
  margin, sequence neighbours checked) and run again, its files and
  results kept apart by margin. This repeats while the margin it names is
  larger than the one it ran at and at most twice `--margin`; beyond that
  the run fails with the block's margin error. Margins can differ between
  blocks because each block owns only its core and its own checks certify
  what it owns. The result records each enlarged block's margin.
- Clashes: both atoms of each owned clash are in the block, at least
  5 Å inside its outer edge.
- Restraints: each owned residue's sequence neighbours i-2..i+2 and every
  atom within 3 Å of its atoms (cctbx's automatic linking distance) are in
  the block. Whole residues are always included.

### Chain segmentation

iotbx starts a new chain object at a change of chain id, at a TER (PDB), or
at a change of `label_asym_id` after an amino acid or RNA/DNA atom (mmCIF).
`read_model` records each residue's chain object (`run_of`), and block files
keep those objects: where a block would put two of them side by side, a TER
(PDB) or a distinct `label_asym_id` (mmCIF) keeps them apart. Two non-polymer
runs of one chain (a ligand, then that chain's waters) may merge, which
changes no result; a polymer residue that would join a preceding run is
refused. reduce joins neighbouring residues of one chain id and ignores TER,
so reduce is given a spare chain id for each stretch (consecutive residues
with one chain id in the model) that a block would otherwise join to another;
its output is restored to the global names before probe and the cliques read it.

### Tools

- reduce must pass the self-check: two inputs from 9A1N, a 12 Å sphere
  around SER 24:276 and the same sphere plus 25 residues 500 Å away, must
  give that serine's hydroxyl the same orientation. Stock reduce (before
  the clique-order patch) does not. Without a pass, block mode is
  refused: only a run asked for as one block (`--block-atoms 0`) proceeds,
  and its result records the failed check.
- probe runs with condensed output (`-CON`), which gives the same clashes
  and overlaps as per-dot output with cctbx's parser (shown on 100k and
  400k atoms). The probe patch for loading speed is optional.

### Deviations from stock MolProbity, recorded in the result

1. reduce with the clique-order patch (its result for a group no longer
   depends on unrelated parts of the model).
2. The NCS shortcut of restraint building is off (before cctbx
   e63952994d it put cysteine SG indices on the wrong atoms).
3. Clash atoms carry the model's names, not the PDB names probe saw.

Occupancy is used as given, as stock MolProbity does;
`--zero-occupancy-as-one` changes it and is recorded. A missing B (`.` or
`?`), which stock cannot read either, is refused unless
`--missing-b-as-zero`, which writes 0 and is recorded; no validation here
uses B.

### Failure

A model iotbx cannot read (`.` or `?` in coordinates, occupancy or B, a
missing atom_site item; in PDB, such a column that is no number), refused
before planning and naming the atom; an unknown residue, double
ownership, block mode with a failed self-check, a block needing a margin
above twice `--margin` (or no larger than the one it ran at), a worker
error, crash or timeout: the run stops and writes no result.

## 4. Resources

- Main process: about 60 bytes per atom in numpy arrays, plus the residue
  table (about 0.5 GB for 8M atoms).
- Worker: about 0.3 GB plus 12 KB per atom of its block, core and margin
  (restraints), plus about 0.5 KB per residue of the model for the
  ownership and stretch maps every worker loads (0.36-0.42 KB measured on
  9A1N subsets); reduce about 1.3 KB per atom and probe about 2.3 KB per
  atom run as subprocesses.
- `--memory GB` chooses the number of workers, not the block size
  (`--block-atoms`, default 130,000): after planning, the most workers
  whose largest planned blocks fit in it together with the main process,
  at least one. The run prints the plan (blocks, overhead, the largest
  block, workers, expected peak) before starting. `--nproc` overrides it;
  a warning is given when the expected peak exceeds the budget. A block run again with a
  larger margin (section 3) can exceed the first round's estimate: each
  is logged with its atoms and expected worker memory, with a warning when
  the round's expected peak exceeds the budget. A hard limit is the caller's
  (the README shows `systemd-run --user --scope -p MemoryMax=...`).
- The budget without `--memory`: the smaller of 90% of what the process's
  cgroup memory limit leaves (cgroup v2 `memory.max`, or `memory.high` if
  lower; v1 `memory.limit_in_bytes`; limit less usage, inactive file cache
  not counted; the tightest of the cgroup and its parents) and 80% of the
  machine's `MemAvailable`. Taken before the model is read, and logged with
  its source. `--memory` replaces it, with a warning if it is more than
  either has. Neither readable: no budget, one worker per CPU. The stock
  subcommand logs the budget too, and warns when its estimate exceeds it.
- Scheduler: one process per block, at most `--nproc` at once, started
  from a forkserver (a worker forked from a process with threads can hang
  on a lock held at fork time); results returned through a pipe; exit
  codes checked, so a worker killed for memory becomes an error naming
  its block; `--block-timeout` turns a hang into an error.
- Files: blocks and reduce and probe scratch in `--workdir` (default: a
  temporary directory beside the output), about 0.1 KB per atom times the
  margin overhead. On success (unless `--keep-workdir`), a workdir the run
  made is removed, and from one it was given only what it made. Kept on a
  failure once the blocks and the plan are written; a run stopped before
  that leaves the directory as it found it. A non-empty workdir without a
  run's plan, and an output inside the workdir, are refused before any
  work.
- Resume: each block result is saved in the workdir as it finishes; the
  plan is saved with a hash of the input and parameters; `--resume` runs
  only the blocks missing.
- Log: the versions first (remolprobity, MolProbity's version and commit,
  cctbx, reduce, probe); one line per block (atoms, time per stage, peak
  memory) and an estimate of the time left; at the end, the time taken, the
  main numbers and an estimate of the peak memory (the main process and the
  blocks running at once, each with its reduce or probe, at their peaks;
  page cache not counted). The provenance
  keeps the time, the peaks of the main process and the largest worker, and
  the cctbx commit; the rest is in the log only.

## 5. Testing

| Tier | What | Where | Time |
|---|---|---|---|
| Unit | `io` (CIF quoting, rows over several lines, `.` and `?`, PDB hybrid-36, multi-model rejection, block file round trip); `blocks` (each residue owned once, every atom within the margin present, deterministic); `result` (canonical order, merge rejects double ownership); adapters checked by IHMValidation 3.3's summary code and 3.4's loader | plain Python, numpy, scipy; GitHub Actions | seconds |
| Component | reduce self-check passes with patched reduce and fails with stock; global name table with 3- and 4-character chain ids; margin checks fail when the margin is too small; regressions: SER 24:276, no degenerate angles, inter-chain bonds made by clashes kept across blocks | MolProbity image | about a minute |
| Oracle | Corpus: 9A1L, 9A1N subsets of 100k, 200k, 400k atoms, and about ten public PDB entries chosen for altlocs, insertion codes, ligands and metals, nucleic acids, many disulfides, explicit hydrogens, more than 9999 residues, long chain ids, zero occupancy (chosen and checked during planning). Oracle 1: stock `molprobity.molprobity` with its converters, same reduce and probe, against one block: counts and identities equal, values within 1e-6; any other difference is a bug or a documented deviation. Oracle 2: one block against blocks of 5k to 20k atoms at several margins (an entry these sizes would not split: blocks of a quarter, an eighth or a half of it, at least 300 atoms; every run of blocks has more than one): identities and counts exact, values within 1e-9, provenance excluded. Oracle 1 also compares stock's own clashscore. References are cached by the model and the image's reduce, probe, het dictionary and cctbx commit; results also by remolprobity's sources | MolProbity image | minutes to an hour |
| Heavy | The 800k subset, one block against blocks, in a 16 GB scope | MolProbity image | hours, on demand |
| Performance | 9A1N in block mode in a 16 GB scope; synthetic assemblies built from public structures; time and memory scaling table | MolProbity image | on demand |

The MolProbity image for tiers 2 to 5 is built from cctbx master with the
reduce and probe patches. Each tier runs with one command.

## Related work

- Proposed changes to cctbx_project: `ramalyze` and `clashscore` counting
  with sets instead of lists (quadratic to linear), and probe's full output
  for Coot run only on demand. `remolprobity stock` and `compare` check
  that they change no result (README, Related work).
- Patch to rlabduke/reduce: clique members sorted by (root, index).
- Patch to probe (repository deprecated): `hisNHcheck` stops at the first
  atom of an earlier residue; carried in the MolProbity image build.
