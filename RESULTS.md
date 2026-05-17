# Results

These are the gem5 measurements for the current CSR-only RVV implementation. Use these numbers as the reported results for the project.

## Scope
- CSR-only sparse kernels
- SpMV: scalar + RVV (accum-style kernel)
- SpMM: scalar + RVV (accum-style kernel)
- Overwrite semantics: y = A x, C = A B

## Environment
- Toolchain: riscv64-linux-gnu-gcc -O2 -march=rv64gcv -mabi=lp64d -static
- Runtime: gem5 RISC-V demo board with RVV enabled
- Vector config: vlen=256
- Vector spec: default v1.0

## Benchmarks
SpMV (bench_riscv):

| Matrix    | Dense | Scalar cyc | RVV cyc | Speedup |
| --------- | ----- | ---------- | ------- | ------- |
| 1000x1000 | 1%    | 160025     | 86673   | 1.85x   |
| 1000x1000 | 5%    | 680806     | 246567  | 2.76x   |
| 1000x1000 | 10%   | 1333663    | 447501  | 2.98x   |

SpMM (bench_spmm_riscv):

| Matrix  | Dense | Bcol | Scalar cyc | RVV cyc | Speedup |
| ------- | ----- | ---- | ---------- | ------- | ------- |
| 256x256 | 1%    | 32   | 249075     | 87337   | 2.85x   |
| 512x512 | 1%    | 32   | 947759     | 304503  | 3.11x   |
| 512x512 | 1%    | 64   | 1790994    | 651971  | 2.75x   |

## Theoretical Speedup

The RVV kernels are accum-style: each vector instruction computes several independent scalar multiply-adds in parallel, then the row or column block is reduced to a scalar result.

For the current `f32` configuration:

$$
VLMAX = \frac{VLEN}{SEW} \times LMUL = \frac{256}{32} \times 1 = 8
$$

That means one RVV `vfmacc` can cover 8 scalar `f32` multiply-adds at once. If the loop is long enough to fill the vector and the memory system is not the bottleneck, the ideal speedup is:

$$
S_{ideal}(N) = \frac{N}{\lceil N / 8 \rceil}
$$

For large `N`, this approaches:

$$
S_{ideal} \to 8\times
$$

How that was derived:

1. The scalar baseline performs one multiply-add per input element.
2. The RVV kernel processes up to 8 elements per vector FMA because `VLMAX = 8` for `f32` with `vlen = 256`.
3. So the vector version reduces the number of arithmetic instructions by roughly a factor of 8.
4. The exact bound for a finite row or block is `N / ceil(N/8)`, which becomes 8 when `N` is large and divisible by 8.
5. The same reasoning applies to the `f64` path in this code because it uses `LMUL = 2`, which still gives 8 lanes on a 256-bit VLEN.

In FLOP terms, each `vfmacc` still performs 2 FLOPs per lane, so the vector kernel can expose up to `16` FLOPs per vector instruction for `f32`. The speedup ceiling, however, is still the lane count, which is `8x`.

## Notes
- These are the gem5 measurements for the current implementation.
- The RVV kernels are accum-style (vector accumulation with a final reduction), not accumulation semantics.
- The theoretical ceiling above is the bound to cite in a proposal; the measured gem5 numbers are the implementation results.

## Reproduce
```sh
make run_test
make run_bench
```
