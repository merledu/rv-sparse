# Benchmark harness

Times each SpGEMM kernel against a reference, checks every result, and appends
one row per run to a CSV.

## Running a sweep

Set `CC` to a compiler that can build and run `-march=rv64gcv`, then run the
preflight. It builds a small vector program to confirm the toolchain and warns
about anything that would make small effects unreliable.

```bash
cp bench/env.sh.example bench/env.sh   # then set CC
bash bench/run_bench.sh --check
```

Fetch the matrices.

```bash
cd matrices && bash getResources.sh && cd ..
```

The full sweep takes hours, so run it in a detachable session. If it stops, run
the same command again and completed configs are skipped.

```bash
mkdir -p bench/results
tmux new -s bench
bash bench/run_bench.sh 2>&1 | tee -a bench/results/run.log
```

Then summarise:

```bash
python3 bench/analyze.py bench/results/spgemm_raw.csv --csv-out bench/results/summary.csv
```

`spgemm_raw.csv` holds one row per timed run and is only ever appended to.
`summary.csv` holds one row per config, with medians and confidence intervals.

Optionally plot it. Both scripts read the summary and need matplotlib.

```bash
python3 bench/plots/speedup_overview.py
python3 bench/plots/speedup_oprow.py
```

Figures are written next to the summary, so `bench/results/`, which is
gitignored. Pass a second argument to send them somewhere else.

To reproduce a published table, run the sweep in one pass as above and
summarise once. Speedups are computed within a single file, so the baseline and
the kernels being scored against it have to be in the same raw CSV.

## Narrowing the run

The whole table runs by default. To run less:

```bash
bash bench/run_bench.sh --kernels contig_f32   # one kernel and its baseline
bash bench/run_bench.sh --kernels intrinsic    # every row with that arm
bash bench/run_bench.sh --dtype f32 --runs 30
```

| Flag | Meaning |
| --- | --- |
| `--check` | preflight only |
| `--check-matrices` | verify canonical CSR over the matrix set, then exit |
| `--dtype` | restrict by dtype |
| `--kernels` | restrict by kernel name or arm |
| `--baseline NAME` | speedup denominator, default `scalar_f32` |
| `--runs N` | timed runs per config, default 10 |
| `--experiments PATH` | alternate experiment table |

Runs resume safely. A config counts as done only at exactly `RUNS` rows, and a
partial group is rerun rather than topped up, so interrupting a sweep is fine.

Narrowing is safe to combine with a full sweep. Every run appends to the same
raw CSV, `--kernels` only chooses which configs execute, and a filter that
excludes the baseline gets it added back so speedups still compute. Several
narrow runs therefore accumulate into one file and one summary.

The way to get this wrong is to set `CSV` or `OUT_DIR` and split runs across
separate files. A summary can only divide by a baseline present in its own
file, so a CSV holding just the vector kernels yields blank speedups. Keep one
raw file unless you are deliberately separating a second toolchain.

## Analysis

`analyze.py` groups by matrix, build, cflags, kernel, and dtype, and reports
median, spread, GOP/s, and a speedup with a bootstrap 95% confidence interval. A
config whose interval includes 1.0 is flagged not significant.

The speedup denominator is the `baseline` arm for the same matrix. Every arm
builds at `rv64gcv`, so the baseline is not a different ISA, it is the same
scalar source with the vectorizer switched off. `autovec` and `baseline` are
then the same code compiled for the same target, and the only thing that varies
is whether the compiler was allowed to vectorize. `--baseline NAME` re-scores
against a different kernel without rerunning anything.

## The experiment table

`experiments.tsv` is the only place experiments are declared, one tab-separated
row per cell of the sweep.

```text
arm        kernel       dtype  build  cflags
baseline   scalar_f32   f32    gcv    -fno-tree-vectorize -fno-tree-slp-vectorize
autovec    scalar_f32   f32    gcv    -
intrinsic  rvv_f32      f32    gcv    -
```

`arm` is what the row is evidence for, `kernel` must match a name in `bench.c`,
`build` is `gc` or `gcv`, and `cflags` is extra compile flags or `-`. An arm is
a kernel and a compilation together, which is why the first two rows share a
kernel and a build and differ only in cflags. Each distinct `(build, cflags)`
combination is its own binary and its own config. A vector kernel on
`build=gc` is rejected before building, since it would not link.

The no-vectorize flags are GCC spelling. Clang ignores `-fno-tree-vectorize`
with a warning rather than an error, so a clang build of that row would be
silently vectorized and stop being a baseline. Use `-fno-vectorize
-fno-slp-vectorize` there.

## Adding a kernel

Build the kernel first, following the strategy guide in
[CONTRIBUTING.md](../CONTRIBUTING.md). Then three edits put it in the sweep.

Register it in `bench.c`, guarding both the wrapper and the registry entry with
`__riscv_vector` for a vector kernel. The last two registry fields are the
validation reference. Point them at a kernel of the same family, and every run
is checked against it, with a mismatch recorded as `correct=0`.

```c
#if defined(__riscv_vector)
KERNEL_WRAP(mykernel_f32_w, rvsp_spgemm_mykernel_f32)
static const kernel_entry_t KERNELS[] = {
    { "mykernel_f32", mykernel_f32_w, RVSP_DTYPE_FP32, "f32",
      scalar_f32_w, "scalar_f32" },
};
#endif
```

Add a row to `experiments.tsv`. If the kernel needs the V extension, add its
name to `vector_kernel()` in `run_bench.sh` so a bad `gc` pairing fails before
the sweep rather than at link time. Then smoke-test before committing to a long
run.

```bash
bash bench/run_bench.sh --kernels mykernel_f32 --runs 2
```

A compile-time tunable is swept with one row per setting, each costing an extra
build.

```text
intrinsic	mykernel_f32	f32	gcv	-DRVSP_MYKERNEL_THRESHOLD=8
intrinsic	mykernel_f32	f32	gcv	-DRVSP_MYKERNEL_THRESHOLD=32
```

## Files

```text
experiments.tsv    the experiment table
run_bench.sh       preflight, build, sweep, resume
bench.c            timing harness, one kernel on one product
csr_check.c        canonical CSR check over the matrix set
analyze.py         raw rows to summary with confidence intervals
plots/             figures from the summary, needs matplotlib
env.sh.example     per-machine config template
results/           CSVs and figures, created on first run
```