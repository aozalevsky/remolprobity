# remolprobity

MolProbity validation of very large atomic models, in spatial blocks.
remolprobity computes Ramachandran, rotamer, clash (with clashscore) and
bond and angle validation for a model of any size. Its result is what stock
MolProbity gives on the whole model's atoms (an atom_site-only copy, as
IHMValidation gives it, without crystal symmetry or header), but its memory
is bounded by the block size, not by the size of the model.

## What it computes, and why

Stock MolProbity (`molprobity.molprobity`, cctbx) processes a model whole.
In the pipeline that IHMValidation runs, its memory grows by about 20 KB per
atom, and parts of it are quadratic. Multi-million-atom integrative models
therefore take hours and tens to hundreds of GB, or fail.

remolprobity runs the same validators in blocks:

- **Blocks.** The model is split into blocks. Each block is a core of
  residues (k-d median splits to a target size) plus a margin: every residue
  with an atom within 25 Å of the core (by default).
- **Ownership.** Each block runs in its own process and keeps only what its
  core owns. A residue owns its Ramachandran and rotamer results. A bond or
  angle belongs to the residue of its first atom, and a clash to the residue
  of its first atom in probe's order.
- **Margin checks.** The plan checks that every block holds the sequence
  neighbours of the residues it owns. As it runs, each block checks that its
  margin holds reduce's hydrogen-bond cliques and the partners of its
  clashes. A block whose margin is too small for these runs again by itself,
  with the margin it asks for, up to twice `--margin`. Beyond that, the run
  stops.
- **Merging.** The block results are merged in a canonical order: residues in
  file order, clashes by atom pair, outliers by atoms. So the result does not
  depend on how the model was split.
- **Only the atoms.** Block files carry the model's atoms and nothing else:
  no crystal symmetry (`CRYST1`, `_cell`, `_symmetry`) and no header. So no
  bond to a symmetry mate is built, and nothing in a header switches on the
  conformation-dependent library for `v33`. This is what IHMValidation's
  atom_site-only copy gives stock MolProbity, and what the tests compare
  with. For a crystal structure whose file has symmetry-related links,
  stock run on the original file can differ.

The tests (see [Testing](#testing)) hold it to two standards:

1. **Stock against one block.** Stock MolProbity and remolprobity running
   the model as one block must agree. Identities and counts are exact and
   values agree within 1e-6, apart from the [deviations](#deviations-from-stock-molprobity)
   below.
2. **One block against many.** Identities and counts are exact, and values
   agree within 1e-9. Values can differ in their last bits because cctbx
   moves each block's coordinates into its own P1 box to build restraints.

It does not compute the rest of MolProbity's report: percentiles, CaBLAM,
C-beta deviations, hydrogen validation, kinemages or Coot scripts. Nor does
it validate against experimental data.

The design, with the reasons behind it, is in [docs/design.md](docs/design.md).

## Requirements

- **MolProbity.** A MolProbity installation built from cctbx
  (`bootstrap.py --builder=molprobity`). Its `mmtbx.python` runs remolprobity,
  which needs nothing beyond cctbx, numpy and scipy.
- **Patched reduce.** The installation's reduce must have the clique-order
  patch, [`patches/reduce-clique-order.patch`](patches/reduce-clique-order.patch).
  - **Why.** Stock reduce lists a clique's members in an order that depends
    on the other cliques. A group's hydrogens can then depend on unrelated
    parts of the model, and blocks would not match a whole-model run.
  - **The self-check.** Every run checks reduce first, on a serine of 9A1N in
    a 12 Å sphere, with and without 25 residues 500 Å away. Its hydroxyl must
    get the same orientation in both.
  - **A reduce that fails it.** Block mode is refused. Only a run asked for
    as one block (`--block-atoms 0`) goes on, and its result records the
    failed check.
- **Patched probe (optional, for speed).**
  [`patches/probe-hisNHcheck-quadratic.patch`](patches/probe-hisNHcheck-quadratic.patch)
  makes probe's `hisNHcheck` stop at the residue's own atoms rather than scan
  every atom loaded before it, which is quadratic in the model. Results are
  unchanged.

To apply both patches, use one of these:

- **A layered image, `container/remolprobity.def`.** It starts from an
  existing MolProbity image and does not rebuild MolProbity. The base image
  (`BASE`, a `.sif` file or a sandbox directory) must hold MolProbity's
  sources under `/opt/molprobity`, as images built with
  `bootstrap.py --builder=molprobity` do, and gcc, g++ and patch on the
  PATH. Build it from the repository root, since its `%files` paths are
  relative to it:

  ```
  TMPDIR=/path/on/disk apptainer build --fakeroot \
      --build-arg BASE=/path/to/molprobity.sif \
      remolprobity.sif container/remolprobity.def
  ```

  The build copies the base image into `TMPDIR`. If `/tmp` is held in
  memory (tmpfs), set `TMPDIR` to a directory on a disk.

  It does the following:
  - keeps the stock reduce at `/opt/remolprobity-stock/reduce`, for the test
    that the self-check fails with it;
  - patches the reduce and probe sources, and rebuilds both with
    `scripts/build_reduce.sh` and `scripts/build_probe.sh`, in place of the
    stock binaries;
  - copies this package to `/opt/remolprobity`, on `PYTHONPATH`, and runs
    it as the image's run script, so `apptainer run remolprobity.sif ...`
    is `mmtbx.python -m remolprobity ...` (see [Usage](#usage)).

  A base that is already patched keeps its stock reduce and its patched
  sources.
- **An existing Apptainer sandbox of MolProbity:** run
  `scripts/dev_image.sh SANDBOX`. It does the same steps in place, except
  that it does not copy the package.
- **Patched binaries built elsewhere:** pass them with `--reduce` and
  `--probe`.

remolprobity is pure Python. To run a checkout rather than the image's copy,
put the checkout on `PYTHONPATH` with `apptainer exec --env PYTHONPATH=...`.
A `PYTHONPATH` given at run time replaces the image's, so the given checkout
is the one imported. To keep the image's copy as well, add `/opt/remolprobity`
to it.

## Usage

```
apptainer run remolprobity.sif MODEL -o RESULT [--geometry v34,v33]
    [--memory GB] [--block-atoms N] [--margin A] [--nproc N]
    [--reduce BIN] [--probe BIN] [--workdir DIR] [--keep-workdir] [--resume]
    [--block-timeout S] [--zero-occupancy-as-one] [--missing-b-as-zero]
```

`apptainer run` on the layered image runs `mmtbx.python -m remolprobity`,
which works the same in any MolProbity installation with the package on
`PYTHONPATH`. For example, in a scope that the kernel holds to 16 GiB, where
the run plans for 90% of what the scope leaves (see [Memory](#memory)):

```
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
    apptainer run remolprobity.sif model.cif -o model.result.json \
    --geometry v34,v33
```

| Option | |
|---|---|
| `MODEL` | One model, mmCIF or PDB, optionally gzipped |
| `-o RESULT` | The result (below), written only when the run succeeds: JSON for a name ending in `.json`, or for `-` (stdout; the log goes to stderr); a pickle for any other name |
| `--geometry` | Bond and angle definitions to compute, comma-separated: `v34` (default), `v33` |
| `--memory GB` | The memory budget for the whole run. It chooses the number of workers. Default: 90% of what the cgroup's memory limit leaves or 80% of what the machine has available, whichever is less (see [Memory](#memory)) |
| `--block-atoms N` | Atoms per block core (default 130,000). `0`: the whole model in one block |
| `--margin A` | Margin in Å around each core (default 25, minimum 10). A block that needs more runs again with what it needs, up to twice this |
| `--nproc N` | Number of workers. Default: as many as the memory budget allows, at most the number of CPUs and of blocks |
| `--reduce`, `--probe` | Binaries to use instead of the installation's |
| `--workdir DIR` | Where block files, scratch and finished block results go. Default: a temporary directory beside the output. See [Workdir](#workdir) |
| `--keep-workdir` | Keep the workdir on success |
| `--resume` | In the workdir of a failed or interrupted run, run only the blocks that have no result. It refuses if the input, the parameters, the tools, the remolprobity version or the cctbx commit changed |
| `--block-timeout S` | Turn a block that hangs into an error |
| `--zero-occupancy-as-one` | Take an occupancy of 0, or none, as 1. Recorded in the result |
| `--missing-b-as-zero` | Take a missing B (`.` or `?` in mmCIF) as 0. No validation here uses B. Recorded in the result |

### Memory

**The budget.** Without `--memory`, a run plans for the smaller of these:

- **90% of what its cgroup's memory limit leaves.** That is the limit less
  what the cgroup already uses, inactive file cache not counted. Of the
  process's cgroup and its parents, the one with the least left counts.
  This is the limit of a `systemd-run` scope, a Slurm job or a container.
  The limit is `memory.max` in cgroup v2 (or `memory.high`, if lower), and
  `memory.limit_in_bytes` in v1.
- **80% of what the machine has available** (`MemAvailable` in
  `/proc/meminfo`). Other jobs can take it during the run, hence the lower
  share.

The log's second line gives the budget and where it came from:

```
memory budget 15.3 GB: 90% of the 17.0 GB left under the 17.2 GB (16.0 GiB) memory limit of cgroup /user.slice/.../run-p1628468-i1530812.scope
memory budget 27.9 GB: 80% of the 34.9 GB the machine has available (no memory limit on its cgroup)
```

`--memory` replaces the budget, with a warning if it is more than the
cgroup's limit leaves or the machine has. Where neither can be read, the
budget is unknown and the run takes one worker per CPU. GB here is 10^9
bytes: systemd's `MemoryMax=16G` is 16 GiB, 17.2 GB.

**The budget is a plan, not a limit.** It chooses the number of workers,
not the block size (that is `--block-atoms`). The run first plans its
blocks, then estimates their costs:

- the main process: 0.6 GB plus 60 bytes per atom of the model;
- each worker: 0.3 GB plus 12 KB per atom of its block, core and margin
  as planned, plus 0.5 KB per residue of the model. Every worker loads
  the whole model's ownership maps; loading them measured 0.36 to
  0.42 KB per residue.

It takes the most workers whose largest blocks, run together, fit within
the budget (at least one). How many atoms a margin adds depends on how
dense the model is, which is why the planned sizes are used.

Before starting, the run prints its plan: blocks, margin overhead, the
largest block, workers and expected peak. It warns if the expected peak
exceeds the budget, and again if blocks that run again with larger
margins may exceed it.

**The hard limit is the scope's.** Run remolprobity in a
`systemd-run --user --scope -p MemoryMax=...` scope, as above, so that the
kernel stops the run rather than the machine. A worker killed for memory
becomes an error naming its block.

### Input

- **One model only.** A multi-model file is refused: split it first.
- **Refused before planning.** These are refused, with an error naming the
  atom; stock MolProbity cannot read them either:
  - in mmCIF, `.` or `?` in coordinates, occupancy or B, or a missing
    `atom_site` item that iotbx needs;
  - in PDB, coordinates, occupancy or B that are not numbers as iotbx reads
    them.

  Other malformed numbers (in mmCIF, a value such as `abc`) may fail with a
  less specific error, or only once a block runs.
- **Refused while a block runs.** Duplicate atoms (the same residue, name
  and altloc twice), as stock refuses them; a block whose names do not fit
  PDB format for reduce and probe, or that has more chain ids than reduce
  can tell apart. The error names the block and gives the message; the
  traceback is kept in the workdir, in
  `results/block#####.traceback.txt`. Any other error in a block carries
  its traceback.
- **Raw PDB-IHM files.** These often give `.` for occupancy and B, and are
  refused, as stock refuses them. 9A1N as distributed is one example. Give
  remolprobity the copy IHMValidation writes for MolProbity, or a filled
  copy, or run with `--zero-occupancy-as-one --missing-b-as-zero`. The
  refusal names the options that would read the atom it stopped at.
- **B is not used.** None of these validations uses B: 4HHB with its B
  values and with every B missing and read as 0 give the same result,
  exactly (a component test).
- **IHMValidation input.** IHMValidation 3.4 writes occupancy 0.0 where the
  deposition gives none. probe ignores atoms with occupancy 0, so such a
  model needs `--zero-occupancy-as-one`. Without it, its clashscore is 0.
  `--zero-occupancy-as-one` also accepts a missing (`.` or `?`) occupancy in
  mmCIF.

### Workdir

- **Default.** A temporary directory beside the output.
- **A directory that does not exist** is made by the run, and removed
  entirely when the run succeeds.
- **An existing empty directory** is used. When the run succeeds, it
  removes only what it made there, so the directory is left empty.
- **A non-empty directory** is refused, unless it holds a run's plan
  (`plan.json`). Then `--resume` continues that run; without `--resume`,
  the run is refused.
- **`-o` inside the workdir** is refused before anything runs, since a
  successful run would remove the result with it.
- **On failure**, the workdir is kept, with its block results and the
  tracebacks of failed blocks, for `--resume`. This holds once the run has
  written its blocks and plan. A run stopped before that, for example by
  the reduce self-check, leaves the directory as it found it: one the run
  made is removed, and one it was given is emptied of what it added.
- **`--keep-workdir`** keeps it when the run succeeds, too.

### Resume and logs

- **Resume.** Each block result is saved in the workdir as soon as the block
  finishes, and the plan is saved with a hash of the input and parameters.
  `--resume --workdir DIR` reuses both.
- **Retried blocks.** A block that ran again with a larger margin keeps its
  files and result apart, under that margin, and `--resume` reuses them
  too.
- **Log.** The log, on stderr, starts with the versions: remolprobity,
  MolProbity (its version and commit, when the installation has
  MolProbity's checkout), cctbx, reduce and probe. The memory budget
  follows (see [Memory](#memory)). Then it has one line per
  block (atoms, time per stage, peak memory) and ends with the time taken,
  the main numbers and an estimate of the peak memory: the main process,
  and the blocks that need the most, as many as run at once, each with its
  reduce or probe, all at their peaks. A memory-limited scope counts more:
  page cache (the image's files, the blocks) and apptainer too. On 4HHB
  the log said 0.45 GB and the scope's peak was 0.64 GB; on large models
  the estimate is the higher, as the peaks do not all come at once. The
  result holds none of this beyond what its provenance already had (see
  [Result](#result)).

  ```
  [remolprobity 17:11:41] remolprobity 0.1.0 with MolProbity 4.5.2 (945670c91f1d), cctbx 2026.10.dev9+65.g9647dd1412, reduce 4.16.250520, probe 2.26.021123
  [remolprobity 17:11:41] memory budget 3.8 GB: 90% of the 4.3 GB left under the 4.3 GB (4.0 GiB) memory limit of cgroup /user.slice/.../run-p1628468-i1530812.scope
  ...
  [remolprobity 17:11:50] done in 9 s: clashscore 142.43, 7 Ramachandran outliers, 49 rotamer outliers
  [remolprobity 17:11:50] memory about 1.52 GB without page cache: main process 0.12 GB; 4 blocks at a time, each up to 0.19 GB, and up to 0.16 GB more for reduce or probe
  ```

### Stock MolProbity, and comparing results

Two more commands help to check results:

- **`stock`** runs stock MolProbity on the whole model and writes its result
  in the same form. This is the reference the oracle tests compare with (see
  [Testing](#testing)). It needs what stock needs, about 20 KB per atom plus
  1 GB, so it is only for models that stock can take. Its log has the
  same versions, memory budget, time and peak memory, and a warning when
  stock's estimate exceeds the budget. The warning does not stop it.
- **`compare`** reads two results, pickle or JSON, and prints a summary of
  each and their differences. It exits with 0 when they are the same, 1
  when they differ and 2 when a file is not a result.

```
apptainer run remolprobity.sif stock model.cif -o stock.json
apptainer run remolprobity.sif model.cif -o one.json --geometry v34,v33 --block-atoms 0
apptainer run remolprobity.sif model.cif -o blocks.json --geometry v34,v33 --block-atoms 1500
apptainer run remolprobity.sif compare stock.json one.json
apptainer run remolprobity.sif compare --tol 1e-9 one.json blocks.json
```

What `compare` checks and prints:

- **Equality.** Identities and counts must be equal, and values within
  `--tol`. The default is 1e-6, the agreement of stock with one block;
  one block and blocks agree within 1e-9. Provenance (times, paths,
  versions) is not compared.
- **Differences.** Each difference names its place in the result, e.g.
  `clashes.records[12].overlap: 0.412 != 0.415`. `--limit` sets how many
  are listed (default 50).
- **Summary.** It lists the headline numbers of both results: residues,
  Ramachandran, rotamers and clashes. Bond and angle outliers are not in it,
  but they are compared.
- **Geometry in one result only.** A definition that only one of the two
  has is left out. `stock` computes both `v34` and `v33`; remolprobity
  computes only what `--geometry` asks for (`v34` by default).
- **`--json`** prints the comparison as JSON: `same`, `differences`, the two
  summaries, and `not_compared`, the definitions left out.
- **Python 3 only.** `compare` does not need cctbx:
  `python3 -m remolprobity compare A B`, with the package on `PYTHONPATH`.

## Result

The result is a dict of plain Python data (dicts, lists, tuples, numbers,
strings), written as a pickle or as JSON, as `-o` asks:

- **In JSON,** tuples are arrays and the keys of `provenance.blocks.retried`
  are strings.
- **Reading.** `remolprobity.result.read(path)` reads either format, telling
  them apart by content (`-` reads JSON from stdin). It turns arrays back
  into tuples and checks the schema and version.

Schema version 1:

- `schema`: `"remolprobity.result"`, `version`: 1.
- `provenance`:
  - the remolprobity version and the cctbx commit;
  - reduce's and probe's versions and SHA-256, and the self-check result;
  - the input's path, SHA-256, and atom, residue and chain counts;
  - the block parameters (`blocks`: `count`, `block_atoms`, `margin`,
    `overhead`, `workers`), and the margin of each block that ran again with
    a larger one (`blocks.retried`, `{block: margin}`);
  - the geometry definitions computed;
  - the deviations from stock MolProbity (below);
  - the options that change data (`--zero-occupancy-as-one`,
    `--missing-b-as-zero`);
  - the wall time and peak memory (`peak_gb`: the main process, and the
    largest worker without its reduce or probe).
- `ramachandran`: `total`, `favored`, `allowed` and `outliers`, plus
  `records`. These are every result as ramalyze returns it, altloc variants
  included: `(chain, resid, resname, altloc, class, phi, psi, score)`.
- `rotamers`: the same counts, plus `records`, each
  `(chain, resid, resname, altloc, evaluation, name, score)`.
- `clashes`: `clashscore`, `atoms` (probe's count, with hydrogens), and
  `records`, each `(atom1, atom2, overlap, distance)`. An atom is
  `(chain, resid, resname, name, altloc)`, in the model's names.
- `geometry`: per definition requested, `bonds` and `angles`, each with a
  `total` and its `outliers`:
  - `v34`, as IHMValidation 3.4 computes it:
    - restraints scored as `mp_validate_bonds` scores them in the installed
      cctbx (covalent geometry and SS BOND on cctbx from 2026-08-19);
    - the conformation-dependent library is used, and symmetric side chains
      are not flipped;
    - outliers are beyond 5 sigma, as `(atoms..., observed, ideal, sigma, z)`;
    - totals exclude hydrogens.
  - `v33`, as IHMValidation 3.3 computes it, with `restraints.combined`:
    - no conformation-dependent library (`molprobity.molprobity` enables it
      only when the file says it was used in refinement), and side chains are
      flipped;
    - outliers are covalent-geometry restraints beyond 4 sigma, as
      `(atoms..., observed, ideal, |z|)`;
    - totals are over covalent-type restraints.
- `residues`: one entry per residue, in file order: `chain`, `resid` and
  `resname`. With `v34`, it also holds the bonds and angles scored and their
  sum of Z² (`v34`: `{'bonds': [n, sum_z2], 'angles': [n, sum_z2]}`). This
  is what IHMValidation 3.4 reports per residue; 3.3 has no per-residue
  geometry.

`resid` is the residue number with its insertion code, if any: `'42'`,
`'42A'`.

## Adapters

The adapters write IHMValidation's MolProbity cache (`<stem>.mp.pkl`) from a
result. They need only Python 3 (no cctbx):

```
python3 -m remolprobity.adapters.ihmv34 RESULT MODEL_ID OUT.mp.pkl \
    [--software KEY=VALUE] [--ihmv-version V]
python3 -m remolprobity.adapters.ihmv33 RESULT MODEL_ID OUT.mp.pkl
```

`--software KEY=VALUE` is repeatable: give it once for each software version
to record.

- **`ihmv34`.** IHMValidation 3.4's cache: its envelope (format, creation
  time, software versions), around the data its `molprobity_convert.py`
  writes.
  - It needs `--geometry v34`.
  - Per-residue entries take the first altloc's Ramachandran and rotamer
    results.
  - Atoms lose their altloc field.
- **`ihmv33`.** IHMValidation 3.3's cache, a bare `{model id: data}`, as the
  code in the 2026 images reads it.
  - It needs `--geometry v33`, and drops clash distances.
  - It differs from 3.3's own converter in names only. Residue numbers are
    decimal: 3.3 writes cctbx's hybrid-36 above 9999, e.g. A000 for 10000.
    Clash atoms carry the model's names, not the PDB names that probe saw.

The adapters take one model, written under `MODEL_ID`. remolprobity refuses
multi-model files, so run each model of a multi-model deposition on its own.

## Deviations from stock MolProbity

The result's `provenance.deviations` lists them (`result.DEVIATIONS`):

1. reduce has the clique-order patch: its result for a group does not depend
   on unrelated parts of the model.
2. Restraints are built without the NCS shortcut. Before cctbx e63952994d,
   that shortcut put cysteine SG indices on the wrong atoms.
3. Clash atoms are named as in the model, not as the PDB names that probe
   saw.

A one-block run with a reduce that fails the self-check leaves out the
first.

Where stock MolProbity itself gets no result, or a different count:

- **Unknown element.** IHMValidation 3.4 with stock MolProbity gets no
  result for a model with an element iotbx does not know, because it fails
  while pickling the validation. 5PTI's UNX with element `X` is one example.
  remolprobity gives the validation's result, and it equals stock's.
- **Hydrogens with microheterogeneity.** Stock `molprobity.molprobity`
  crashes in its hydrogen validation (`validate_H`) on models that have
  both, as 1EJG does. remolprobity does no hydrogen validation, and equals
  stock in everything else.
- **First conformer B.** Where a chain's first conformer is B, ramalyze can
  count a residue that it does not list. remolprobity counts from the
  records. No model in the test corpus has this case.

### Engines

remolprobity uses reduce, probe and clashscore, not series 2 (reduce2,
probe2, clashscore2). Series 2 needs more memory per atom: reduce2 about
22 KB and clashscore2 about 54 KB, against about 27 KB for series 1.
reduce2's hydrogens also depend on atoms far away; a fix is proposed in
[cctbx/cctbx_project#1245](https://github.com/cctbx/cctbx_project/pull/1245).

## Testing

| Tier | Command | What it needs |
|---|---|---|
| Unit | `python3 -m pytest tests/unit -q` | Python 3.10+, numpy, scipy, pytest (runs on the host, and in GitHub Actions) |
| Component | `REMOLPROBITY_IMAGE=IMAGE python3 -m pytest tests/component -q` | A MolProbity image with the patched reduce, and the stock one at `/opt/remolprobity-stock/reduce` (or `REMOLPROBITY_STOCK_REDUCE`): the layered image, or a sandbox prepared by `scripts/dev_image.sh`; a few minutes |
| Oracle | `scripts/fetch_corpus.py`, then `REMOLPROBITY_IMAGE=IMAGE python3 -m pytest tests/oracle -m oracle` | The corpus, and runs of stock MolProbity in memory-limited scopes (below); minutes to an hour |
| Heavy | `scripts/fetch_corpus.py --tier heavy`, then `REMOLPROBITY_IMAGE=IMAGE python3 -m pytest tests/heavy -m heavy` | Two runs of the 800k-atom subset of 9A1N, each in a 16 GB scope; about half an hour |

- **Default run.** `python3 -m pytest` runs the unit and component tiers. The
  component tests are skipped without `REMOLPROBITY_IMAGE`.
- **cctbx version.** `test_glycan_link_restraints_are_not_scored` expects
  the bond total of the installed cctbx:
  - 61 on cctbx from 2026-08-19 on, which scores only covalent geometry and
    SS BOND restraints;
  - 65 before, with the glycan links scored.

  It tells them apart as `geometry.py` does: by whether
  `mp_validate_bonds._scoreable_origin_ids` exists. In both cases it checks
  that remolprobity scores what stock does.
- **IHMValidation adapter tests.** These run IHMValidation's own code on the
  caches. They are skipped unless `REMOLPROBITY_IHMV_IMAGE` (an
  IHMValidation 3.3 image) and `REMOLPROBITY_IHMV34` (a 3.4 checkout) are
  set.
- **The corpus.** It is listed in `tests/corpus/corpus.json`: public PDB
  entries chosen for their edge cases, plus subsets and tiles of 9A1N. It is
  kept in `~/.cache/remolprobity/corpus` (or `REMOLPROBITY_CORPUS`), with the
  oracle's cached references and results.
- **The oracle tiers.**
  - Oracle 1 runs stock MolProbity (`remolprobity stock`) on entries of up
    to 450,000 atoms and compares it with remolprobity in one block, stock's
    own clashscore included.
  - Oracle 2 compares one block with blocks of 5,000 and 20,000 atoms at
    margins of 15 to 25 Å. An entry that one of these sizes would not
    split gets blocks of a quarter, an eighth or a half of it instead (at
    least 300 atoms), so every entry is compared across layouts.
  - **Caches.** Stock's references are kept by the model's SHA-256 and by
    the image's reduce, probe and het dictionary hashes and cctbx commit.
    remolprobity's results are kept by those and by the SHA-256 of its
    sources, so a change to the code runs them again.
  - **Workers and memory.** `REMOLPROBITY_ORACLE_NPROC` sets the workers
    of a run of blocks (default 4). `REMOLPROBITY_ORACLE_MAX_GB` (default
    4) is the scope of each run, and any run expected to need more is
    skipped, with what it would need. The estimate is remolprobity's own
    for the run's plan, or 20 KB per atom plus 1 GB for stock.

## Performance

remolprobity in block mode, with `--memory 16` and the default block size
and margin, each run in a 16 GB scope by `scripts/perf.py`:

- **Machine.** A shared workstation (48 CPUs, 62 GB) running other jobs at
  the same time.
- **Peak.** The scope's `memory.peak`, page cache included.
- **Wall time.** Includes starting the image.
- **Inputs.** `scripts/fetch_corpus.py --tier perf` makes the subsets and
  tiles. The whole-model row uses `9a1n_filled`: 9A1N with occupancy 1.00 and
  B 0.00 filled in, because the file as distributed has neither.
- **Workers.** These rows were measured when `--memory` sized every worker
  as 1.4 times the core size. Workers now come from the planned blocks
  (see [Memory](#memory)). The same plans now get 1 worker for the 100k
  subset (expected 2.1 GB), 4 for the 400k subset (6.9 GB) and 8 for the
  800k subset (14.1 GB).

| Model | Atoms | Wall time (min) | Peak (GB) | Blocks | Workers | Margin overhead | Blocks run again |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9A1N, 100k subset | 101,764 | 1.4 | 1.5 | 1 | 1 | 1.00 | 0 |
| 9A1N, 400k subset | 401,984 | 1.7 | 5.4 | 4 | 4 | 1.03 | 0 |
| 9A1N, 800k subset | 800,464 | 3.3 | 8.0 | 8 | 6 | 1.11 | 0 |
| 9A1N | 1,893,744 | not yet measured | | | | | |
| 9A1N, 2 copies | 3,787,488 | not yet measured | | | | | |
| 9A1N, 4 copies | 7,574,976 | not yet measured | | | | | |

To run one model:

```
REMOLPROBITY_IMAGE=IMAGE scripts/perf.py MODEL --memory 16 --out DIR
```

It prints one JSON line.

## Related work

remolprobity is also a test bed for changes to cctbx proposed in
[cctbx/cctbx_project#1245](https://github.com/cctbx/cctbx_project/pull/1245)
(issue [#1246](https://github.com/cctbx/cctbx_project/issues/1246)):

- **ramalyze** counts with sets instead of lists, which makes it linear
  rather than quadratic in the number of residues. Its results stay as they
  were.
- **reduce2's Optimizer** becomes linear in the number of Movers, and its
  hydrogens independent of atoms far away.

remolprobity needs neither: its blocks are small, and it uses reduce, not
reduce2 (see [Engines](#engines)).

To check on a model that the ramalyze change leaves every result as it was,
run `remolprobity stock` on it with and without the change, and
`remolprobity compare` the two results (see
[Stock MolProbity, and comparing results](#stock-molprobity-and-comparing-results)).
The time and peak memory in the two logs show what it saves. For reduce2,
the self-check's pair of models (see [Requirements](#requirements)),
`remolprobity/data/selfcheck_near.pdb` and `selfcheck_far.pdb`, show the
dependence: with reduce2 on cctbx master, 2 of the sphere's 235 hydrogens differ
between the two; with the change, none do.

## License

remolprobity is under the [MIT License](LICENSE), with two exceptions:

- **The patches** in `patches/` change reduce and probe, from the
  Richardson Lab at Duke University, which are under the Apache License
  2.0. The patches are under the same license
  ([`patches/LICENSE`](patches/LICENSE), [`patches/NOTICE`](patches/NOTICE)),
  so that they can go upstream.
- **Test data** in `tests/data` and `remolprobity/data` are coordinates
  from PDB entries 1CRN, 1EHZ, 4HHB and 9A1N, or parts of them. PDB data
  are free of copyright restrictions (CC0 1.0).

remolprobity includes none of the software it runs on. cctbx is under a
BSD-style license from Lawrence Berkeley National Laboratory. MolProbity,
reduce and probe are under the Apache License 2.0. An image built with
`container/remolprobity.def` holds all of them, each under its own
license.
