# rv-sparse: RISC-V Vector Accelerated Sparse Linear Algebra

This repository is developed for the **LFX Summer 2026 Mentorship** under the **Micro Electronics Research Laboratory (MERL)**. The goal of this project is to provide a lightweight, highly optimized C library for sparse matrix operations utilizing **RISC-V Vector (RVV) Intrinsics**.

## Architectural Highlights
Because sparse matrix operations are historically memory-bound, this implementation focuses heavily on **Hardware-Software Co-design** and cache efficiency:

1. **Vector-Length Agnostic (VLA)**: All RVV kernels use `__riscv_vsetvl_e32m1` strip-mining to ensure the binaries can run on any conformant RISC-V vector processor, regardless of the underlying hardware vector register width (VLEN).
2. **SpMM Zero-Gather Optimization**: Sparse Matrix-Matrix Multiplication (SpMM) notoriously suffers from irregular memory accesses. By changing the axis of vectorization to stride across the columns of the *dense* matrix, we eliminate the need for expensive vector gather/scatter instructions. This results in purely contiguous `vle32` and `vse32` memory operations, maximizing cache line utilization.
3. **Register Pressure & Footprint**: We enforce `int32_t` for CSR indices. Unlike 64-bit size types which halve vector register occupancy and double memory traffic, our 32-bit index design ensures we can pack maximum data into `vint32m1_t` registers.
4. **Memory Alignment**: Core structures enforce native cache-line alignment to avoid unaligned vector load penalties during traversal.

## Current Implementations
* **Data Structures**: `rv_sparse.h` defines cache-aware CSR and CSC structs.
* **Scalar Fallbacks**: Clean C reference implementations for validation.
* **SpMV RVV Kernel**: Uses indexed loads (`vloxei32`) and fast unordered vector reduction (`vfredusum`).
* **SpMM RVV Kernel**: Uses contiguous strip-mining over dense columns, completely avoiding scatters/gathers.

## Build Instructions (Cross-Compilation)
We support building via the `riscv64-elf-gcc` bare-metal toolchain. The CMake configuration automatically bypasses AppleClang injection on macOS and generates static RISC-V archives (`.a`) using `-march=rv64gcv`.

```bash
mkdir build && cd build
cmake ..
make
```

## Next Steps
- Implement QEMU/Spike integration for automated numerical correctness validation.
- Implement CSR to CSC format conversions.
- Extensive micro-benchmarking using hardware performance counters (measuring IPC, cache misses).
