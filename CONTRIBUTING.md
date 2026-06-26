# Contributing Guidelines

First off, thank you for considering contributing to `rv-sparse`.

This project is under active development. Our goal is to build a clean sparse linear algebra library with a stable public API, while also providing a separate space for experimental kernels, benchmarks, and architecture-specific optimizations.

Please follow the guidelines below when proposing changes, adding new features, optimizing kernels, or submitting benchmark results.

## Contribution Workflow

We follow a strict feature-branch workflow.

Do not commit directly to `main`.

All changes should be submitted through Pull Requests.

## 1. Sync Your Local Environment

Before starting new work, make sure your local repository is up to date.

If you are working directly from the main repository:

```bash
git checkout main
git pull origin main
```

If you are working from a fork:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## 2. Create a Development Branch

Create a new branch for your work.

Use the following naming convention:

```text
dev_YOUR_FEATURE
```

Keep the feature name concise but descriptive.

Examples:

```bash
git checkout -b dev_rvv_intrinsics
git checkout -b dev_spgemm_i8_kernel
git checkout -b dev_spmv_optimization
git checkout -b dev_benchmark_harness
git checkout -b dev_update_docs
```

Maintainers may also use internal integration or review branches such as:

```text
experiment/raw-kernels-rvv
review/<contributor-name>-<topic>
```

Examples:

```text
experiment/raw-kernels-rvv
review/ayokunle-benchmarks
review/talha-rvv-kernel
```

## 3. Make Atomic Commits

Use small, logical commits. Each commit should represent one clear change.

We recommend Conventional Commits:

```text
feat: add a new feature
perf: improve performance
fix: fix a bug
docs: update documentation
test: add or update tests
bench: add or update benchmarks
refactor: restructure code without changing behavior
```

Examples:

```bash
git commit -m "perf: add experimental RVV INT8 row accumulation kernel"
git commit -m "test: compare RVV kernel against scalar reference"
git commit -m "bench: add raw SpGEMM benchmark harness"
git commit -m "docs: update raw kernel experiment README"
```

Avoid vague commit messages such as:

```text
update
fix stuff
changes
work
```

## 4. Push Your Branch

Push your development branch to your fork or to the appropriate remote repository:

```bash
git push -u origin dev_YOUR_FEATURE
```

Example:

```bash
git push -u origin dev_rvv_intrinsics
```

## 5. Open a Pull Request

Open a Pull Request from your development branch.

The PR description should include:

1. Motivation: why this change is needed.
2. Implementation details: brief explanation of the technical approach.
3. Testing commands: exact commands used to build and test.
4. Benchmark results, if performance is affected.
5. Whether the public API is changed.
6. Target architecture or emulator used, if applicable.

For stable library changes, open the PR against:

```text
main
```

For experimental raw kernels, benchmarks, or early optimization work, open the PR against the active experimental branch when available:

```text
experiment/raw-kernels-rvv
```

Do not target `main` for experimental kernels unless they are already validated and approved for integration.

## Public API Policy

The public API must remain stable.

Public API files include:

```text
include/rv_sparse.h
include/rv_sparse_types.h
```

Do not modify the public API for experimental kernels unless the change has been discussed and approved.

Experimental kernels should first be developed outside the public API path.

## Experimental Kernel Work

Experimental kernels should be developed in:

```text
experiments/raw_kernels/
```

This directory is used for raw kernels, correctness tests, early benchmarking, and architecture-specific experiments.

Experimental kernels must not be exposed through the public dispatcher until they are:

1. Correct.
2. Tested against a scalar reference.
3. Validated on native builds.
4. Validated on RISC-V/QEMU when applicable.
5. Benchmarked.
6. Reviewed.

Only selected and validated kernels should later be migrated into:

```text
src/kernels/spgemm/
```

and connected through the wrapper and dispatcher layers.

## Current Kernel Development Target

The initial experimental optimization target is:

```text
INT8 x INT8 -> INT32
```

This target is used to evaluate:

```text
scalar kernels
unrolled kernels
GCC auto-vectorized kernels
RISC-V Vector intrinsics kernels
```

## RISC-V Vector Intrinsics

RISC-V Vector kernels should use the C intrinsics interface through:

```c
#include <riscv_vector.h>
```

RVV-specific code should be guarded with appropriate compile-time checks when needed.

Example:

```c
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif
```

Intrinsic-heavy code should include comments explaining:

1. The vector type used.
2. The widening or narrowing operation.
3. The memory access pattern.
4. Any assumptions about alignment or index validity.
5. Any limitations, such as duplicate column handling.

## Code Organization

The main library follows a layered structure:

```text
Public API
Dispatch layer
Internal wrapper layer
Raw kernel layer
```

Raw kernels should use pointer-based interfaces and should not depend directly on public matrix structs unless they are part of the integration layer.

For experimental kernels, prefer the following structure:

```text
experiments/raw_kernels/
├── kernels/
├── tests/
├── benchmarks/
└── README.md
```

## Tests

Every new kernel or backend should include a correctness test.

Tests should compare against a known reference implementation whenever possible.

For main library tests, use:

```text
tests/
```

For experimental kernels, use:

```text
experiments/raw_kernels/tests/
```

Before opening a Pull Request, run:

```bash
make clean
make
make test
```

For RISC-V/QEMU validation, run:

```bash
make clean
make test TARGET_ARCH=riscv
```

For experimental raw kernels, run from the experimental directory:

```bash
cd experiments/raw_kernels
make clean
make test
make clean
make test TARGET_ARCH=riscv
```

## Benchmarks

Benchmark contributions must be reproducible.

A benchmark contribution should include:

1. Source code.
2. A README or usage instructions.
3. The exact command used to run the benchmark.
4. The metrics reported.
5. The tested architecture, compiler, and flags.
6. Whether the benchmark was run on native hardware, QEMU, or another emulator.

Useful metrics include:

```text
kernel name
dtype
matrix size
density
nnz(A)
nnz(B)
nnz(C)
execution time
GOPS or GFLOPS
speedup against scalar baseline
memory footprint
correctness status
compiler version
compile flags
target architecture
```

For INT8 kernels, use GOPS when appropriate.

For FP32 kernels, use GFLOPS when appropriate.

Do not commit large datasets, generated binaries, or heavy raw benchmark outputs unless explicitly approved.

## GCC Auto-Vectorization Reports

If a contribution discusses GCC auto-vectorization, include evidence.

Recommended compiler flags:

```bash
-fopt-info-vec
-fopt-info-vec-missed
```

When needed, also provide assembly output:

```bash
-S
```

For RISC-V Vector auto-vectorization, report:

```text
GCC version
compile command
-march and -mabi flags
vectorization report
whether RVV instructions appear in the generated assembly
```

## Files That Should Not Be Committed

Do not commit generated artifacts such as:

```text
bin/
obj/
lib/
build/
*.o
*.a
*.so
*.exe
*.out
```

Do not commit local personal notes such as:

```text
docs/personal_notes.md
```

Do not commit local benchmark executables such as:

```text
bench_f32
bench_i8
bench_unroll4
```

Large datasets and raw benchmark outputs should be avoided unless explicitly approved.

## Authorship and External Contributions

Do not copy another contributor's work into your own branch manually.

To preserve authorship, use Pull Requests, remotes, or merges from the original contributor branch.

When reviewing another contributor's work, create a review branch:

```bash
git fetch <remote-name>
git checkout -b review/<contributor-name>-<topic> <remote-name>/<branch>
```

Example:

```bash
git fetch ayokunle
git checkout -b review/ayokunle-benchmarks ayokunle/main
```

After reviewing and testing, merge only if the contribution is correct, clean, and properly attributed.

## Pull Request Checklist

Before requesting a review, make sure:

* [ ] I have followed the `dev_YOUR_FEATURE` branch naming convention.
* [ ] My code compiles without warnings on the target architecture.
* [ ] I have run the relevant correctness tests.
* [ ] I have run RISC-V/QEMU tests when applicable.
* [ ] I have compared new kernels against a scalar reference.
* [ ] I have included benchmark results if performance is affected.
* [ ] I have documented the benchmark commands and environment.
* [ ] I have performed a self-review of my own code.
* [ ] I have commented low-level or hard-to-understand code.
* [ ] I have updated README files or documentation when applicable.
* [ ] I have not committed generated artifacts, binaries, build folders, or large datasets.
* [ ] I have preserved authorship for external contributions.

## Code Review Process

Once your Pull Request is open, assign it to a maintainer or request review from the appropriate contributor.

The review will check:

1. Correctness.
2. Code organization.
3. API stability.
4. Performance impact.
5. Benchmark reproducibility.
6. RISC-V compatibility when applicable.
7. Authorship and commit cleanliness.

Requested changes should be added as new commits to the same development branch.

Once approved, the maintainer may merge using either:

```text
Squash and merge
```

for small feature branches, or:

```text
Create a merge commit
```

when preserving individual commit authorship is important.

## Integration Policy for Optimized Kernels

Experimental kernels should not be integrated into the public API immediately.

The recommended path is:

```text
experiments/raw_kernels/
benchmark and compare
select best kernel
migrate to src/kernels/spgemm/
add wrapper
add dispatcher support
add public tests
update documentation
open PR to main
```

Only the best validated kernel should be migrated to the main backend.

## License

By contributing, you agree that your contributions are provided under the same license as the project.
