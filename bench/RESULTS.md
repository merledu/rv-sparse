# Baseline measurements

Numbers behind the synthetic sweep in `run_bench.sh`. Everything here was
measured with `scalar_f32` on x86-64 before any tiling work, so it records the
problem tiling is meant to solve, not a tiling result.

Re-measure on the target board before drawing conclusions. The shape of these
curves should carry over; the absolute numbers will not.

## Why the accumulator is the interesting variable

All kernels accumulate into a dense array indexed by column of B. `rvsp_ws_bytes`
allocates 13 bytes per column of that array — 4 for the value, 1 for the mark,
and 4 each for the touched list and the sort scratch. The array is sized by the
matrix width alone, so a wider matrix means a bigger accumulator even if the
number of nonzeros stays the same.

Once the accumulator no longer fits in cache, every `acc[col]` and `mark[col]`
probe becomes a memory access, and the kernel stops being limited by arithmetic.

## Cache ladder

Nonzeros per row is pinned at 8 across every row of the table, so the work per
row is constant and the only thing changing is the accumulator footprint. The
`d8` entries in `GEN_CASES` are these cases.

| cols | accumulator | nnz(A) | nnz(C) | time | GOP/s | step |
| --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 52 KB | 32,729 | 263,029 | 0.0044 s | 0.1194 | — |
| 16,384 | 208 KB | 131,305 | 1,048,836 | 0.0183 s | 0.1150 | 1.04x |
| 65,536 | 832 KB | 525,344 | 4,209,245 | 0.1029 s | 0.0818 | 1.41x |
| 262,144 | 3.3 MB | 2,097,552 | 16,775,532 | 0.7082 s | 0.0474 | 1.73x |
| 1,048,576 | 13 MB | 8,388,376 | 67,111,891 | 3.6802 s | 0.0365 | 1.30x |

Throughput falls 3.27x from top to bottom. Work per row is identical throughout,
so the loss is entirely accumulator locality.

Two features of the curve matter. The knee sits between 208 KB and 832 KB, which
is where the accumulator stops fitting in L2 on this machine. And the penalty
saturates: the step from 3.3 MB to 13 MB costs only 1.30x, against 1.73x for the
step before it. Once the accumulator is comfortably larger than cache, making it
larger still adds little, because the probes are already paying full memory
latency.

The practical consequence is that the interesting region for tiling begins
around the knee, and a tiled kernel that restores the top-row throughput would
be worth about 3.3x at 1M columns. That figure is the upside the tiling work is
chasing, and it is what the eventual tiled numbers should be compared against.

This is the effect a tiled kernel has to recover. It also means a tiling
experiment run only on the original `GEN_CASES` sizes — all of which are at or
below the 4,096 row of this table — would measure nothing, because at that size
there is no locality problem to fix.

## Two generator bugs found while building the ladder

### Requested density was overshot 3.4x to 4.5x

`genmat.c` drew row degrees from a lognormal whose sigma was computed as

```c
sqrt(log(1 + std * std / avg * avg))
```

C precedence makes `std * std / avg * avg` equal to `std * std`, not the
intended `(std/avg)^2`. With the default `cv = 0.5` that inflates sigma from
0.47 to 1.68, and since a lognormal's mean is `exp(mu + sigma^2/2)`, the
generated mean degree came out roughly `exp((1.68^2 - 0.47^2)/2) = 3.7x` too
high.

The `vals_pos = avg - 3 * std` branch selects the lognormal path for any
`cv > 1/3`, so the default configuration always hit it.

| N | requested density | expected nnz | before fix | after fix |
| --- | --- | --- | --- | --- |
| 4,096 | 0.001953 | 32,766 | 117,768 (3.59x) | 32,729 (1.00x) |
| 16,384 | 0.000488 | 130,997 | 502,997 (3.84x) | 131,305 (1.00x) |
| 2,048 | 0.005 | 20,972 | 94,450 (4.50x) | 21,036 (1.00x) |
| 2,048 | 0.05 | 209,715 | 719,534 (3.43x) | 204,252 (0.97x) |
| 4,096 | 0.01 | 167,772 | — | 166,978 (1.00x) |

Fixed by parenthesising the division. Density now lands within 3% of the
request across the whole range.

This mattered beyond tidiness: a sparsity knob that misses by a factor of four
cannot express "95% sparse" at all, and the overshoot also inflated the working
set, which is what made the largest cases look unaffordable.

**Any generated-matrix rows collected before this fix are not comparable to rows
collected after it.** Purge `gen_*` labels from an existing
`results/spgemm_raw.csv` rather than appending to them. Real-matrix rows are
unaffected.

### Generation is O(N^2) regardless of density

Each distribution function allocates an `is_nz_ind` array of N entries and
zeroes all N of them once per row, then scans a full bandwidth-wide range to
collect the marked columns. Both are O(N) per row, so generating a matrix costs
O(N^2) no matter how sparse it is.

| N | generation time | ratio for 2x size |
| --- | --- | --- |
| 65,536 | 2.37 s | — |
| 262,144 | 35.71 s | 15.1x (for 4x) |
| 524,288 | 166.44 s | 4.66x |
| 1,048,576 | 614.27 s | 3.69x |

A ratio near 4 for each doubling is the quadratic signature.

At N = 1,048,576 that is 10.2 minutes per matrix, and `bench.c` generates both A
and B, so roughly 20 minutes of setup precedes the first multiply. The multiply
itself is far cheaper. This is not fixed here, because the fix is a real change
to the generator's sampling strategy rather than a typo, but it dominates the
cost of the largest cases and should be addressed before the 1M case is used in
a sweep. Two options, in increasing order of effort:

* Generate once and cache to disk in a binary CSR format, so the cost is paid
  once instead of once per config.
* Replace the per-row scratch sweep with direct sampling of `degree` distinct
  columns, making generation O(nnz).

## Structure knobs

`--gen` now exposes the generator's structure parameters, which were previously
unreachable because `bench.c` called `genmat_default_params` and overrode only
density and seed. Measured at 2,048 x 2,048, density 0.005:

| variant | nnz(A) | nnz(C) | op_max | op_var |
| --- | --- | --- | --- | --- |
| uniform | 94,450 | 1,320,465 | 70,373 | 4.16e7 |
| `--gen-cv 2.5` | 108,216 | 1,126,654 | 78,069 | 7.96e7 |
| `--gen-band 64 64` | 44,511 | 225,598 | 4,416 | 7.82e5 |

(Collected before the density fix, so treat the absolute nnz as indicative. The
contrast between variants is the point.)

High `cv` nearly doubles the variance of per-row work, which is the load
imbalance case. Banding cuts `op_max` by 16x and gives columns real locality,
which should make it the case where tiling helps least — a useful control,
because uniform-random matrices have no column locality at all and will flatter
a tiling result.

Note that banding also reduces nnz for a given density, since it restricts where
nonzeros may fall. Comparing a banded case against an unbanded one at equal
density compares different amounts of work.

There is a second reason to care about banding. `contig_f32` returns exactly the
`scalar_f32` result on uniform-random matrices at every tile size, meaning its
contiguous-run path never fires: random sparsity produces no runs of
`RVSP_CONTIG_MIN` (default 8) adjacent columns. The contig arm measures nothing
on the unbanded generated cases, and needs banded or real matrices to be
meaningful.

## Tiling: RVSP_TILE_COLS

`RVSP_TILE_COLS` walks B in column strips so the live slice of the accumulator
stays resident. 0 disables it and restores the original full-width pass. Only
the numeric pass is tiled so far; the two symbolic passes still run full width,
so roughly two thirds of the scattered traffic is untouched.

### Correctness on RVV

Validated under `qemu-riscv64 -cpu rv64,v=true`, since none of the vector
kernels had been exercised with tiling on.

`make test TARGET_ARCH=riscv` passes at `RVSP_TILE_COLS` = 1, 2, 3, 4, 16 and
1024, across all four kernels. The small widths are the useful ones: the
fixtures are a few columns wide, so a tile of 1 puts every column in its own
strip and maximally exercises the boundary and cursor logic. The expected
results in `tests/csr_fixtures.h` are hand-written constants, which is what
makes this a real check -- bench's `correct=` flag validates `scalar_f32`
against `scalar_f32` from the same build and structurally cannot see a tiling
bug.

Cross-build hashes at 512x512 and 2048x2048, from separately compiled binaries
at tile 0, 16 and 1024: `row_ptr` and `col_idx` are bit-identical for every
kernel at every tile size. Structure is unaffected, as intended.

Values are bit-identical for `scalar_f32`, `rvv_f32` and `contig_f32`.
`adaptive_f32` differs, by 1.0 ULP on 0.09% of entries at 2048x2048 -- pure
reassociation, explained below.

### Tiling changes vectorisation decisions

`adaptive_f32` gathers only when a segment holds at least `RVSP_GATHER_MIN`
(default 16) nonzeros. A 16-column strip of a sparse row almost never holds 16
nonzeros, so at `RVSP_TILE_COLS=16` the gather path never fires and the kernel
degrades to scalar. Its checksum at that tile size is exactly the `scalar_f32`
checksum, which is how the mechanism showed up.

So tile width and `RVSP_GATHER_MIN` are coupled: a narrow tile silently
switches the vector path off. A tile-size sweep that holds `RVSP_GATHER_MIN`
fixed is partly measuring lost vectorisation rather than locality, and the two
knobs need sweeping together.

`rvv_f32` is immune because it always gathers, with no length-dependent
dispatch.

### Performance is currently a regression

Measured on x86-64, `scalar_f32`, interleaved repeats:

| N | tile | median GOP/s |
| --- | --- | --- |
| 65,536 | 0 (full width) | 0.0830 0.0830 0.0749 |
| 65,536 | 32,768 | 0.0673 0.0742 0.0750 |
| 65,536 | 131,072 | 0.0560 0.0617 0.0576 |

Untiled wins at every tile size. The diagnostic row is 131,072: that is a
single strip wider than the matrix, so it performs the same arithmetic in the
same order as the untiled pass, yet runs ~27% slower. That isolates the cost to
tiling overhead rather than to any cache effect.

The cause is the per-strip cursor advance, which visits every row of B on every
strip while the useful work is only nnz(B):

| case | cursor steps | nnz(B) | overhead |
| --- | --- | --- | --- |
| N=65,536, tile=2,048 (32 strips) | 2.1M | 525K | 4x |
| N=262,144, tile=2,048 (128 strips) | 33.5M | 2.1M | 16x |
| N=262,144, tile=32,768 (8 strips) | 2.1M | 2.1M | 1x |

That matches the observed shape: smaller tiles get monotonically worse. Making
tiling competitive requires the cursor work to be O(nnz(B)) rather than
O(strips x rows(B)), which is a restructure of the inner loop, not a tweak.

An earlier sweep appeared to show large tiles beating untiled by ~23%. Those
numbers did not reproduce under interleaved repeats and are discarded. The
dev box carries background load and does not pin the governor, and the untiled
arm alone varies 10% run to run, so it cannot resolve effects of this size.
Tile-size measurements belong on the board.

## Reproducing

```bash
make
bench_bin --kernel scalar_f32 --gen 65536 65536 0.000122 42 --runs 3 --warmup 1
```

The ladder is the four `d8` rows of `GEN_CASES`. Structure variants are the
`band` and `cv25` rows.

Tiling correctness under QEMU, for one tile size:

```bash
make test TARGET_ARCH=riscv BUILD_TYPE=debug \
    ARCH_FLAGS="-march=rv64gcv -mabi=lp64d -static -DRVSP_TILE_COLS=16" \
    OBJ_DIR=obj/t16 LIB_DIR=lib/t16 BIN_DIR=bin/t16
```

`BUILD_TYPE=debug` avoids the tracked `-flto` failure in `accum_rvv_f32.c` on
GCC 15. Use a per-tile-size `OBJ_DIR` so objects are not reused across builds.
