// bench/benchmark.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <riscv_vector.h>
#include "rv_sparse.h"

static inline uint64_t rdcycle(void)
{
#if defined(__riscv)
    uint64_t c;
    __asm__ __volatile__("rdcycle %0" : "=r"(c));
    return c;
#else
    return 0;
#endif
}

#define WARMUP 3
#define RUNS 10

typedef void (*spmv_fn)(const CSRMatrix_f32 *A, const float *x, float *y);

static uint64_t time_cycles(spmv_fn fn, const CSRMatrix_f32 *A, const float *x, float *y)
{
    for (int i = 0; i < WARMUP; i++)
        fn(A, x, y);

    uint64_t t0 = rdcycle();
    for (int i = 0; i < RUNS; i++)
        fn(A, x, y);

    return (rdcycle() - t0) / RUNS;
}

static size_t spmv_vlmax_f32(void)
{
#if defined(__riscv)
    return __riscv_vsetvlmax_e32m1();
#else
    return 0;
#endif
}

static void bench_config(int rows, int cols, float density)
{
    CSRMatrix_f32 *A = generate_random_csr_f32(rows, cols, density, 42);
    float *x = malloc(cols * sizeof(float));
    float *y = malloc(rows * sizeof(float));
    for (int i = 0; i < cols; i++)
        x[i] = (float)i * 0.01f;

    uint64_t scalar_cycles = time_cycles(spmv_csr_scalar_f32, A, x, y);

    uint64_t rvv_cycles = time_cycles(spmv_csr_rvv_f32, A, x, y);
    double speedup = (rvv_cycles == 0) ? 0.0 : (double)scalar_cycles / rvv_cycles;
    size_t vlmax = spmv_vlmax_f32();

    printf("%5dx%-5d | %4.0f%% | %6zu | %12lu | %12lu | %5.2fx\n",
           rows, cols, density * 100, vlmax, scalar_cycles, rvv_cycles, speedup);

    // free(x);
    // free(y);
    // csr_free_f32(A);
}

int main(void)
{
    printf("%-12s | %5s | %6s | %12s | %12s | %6s\n",
           "Matrix", "Dense", "VLMAX", "Scalar cyc", "RVV cyc", "Speedup");
    printf("-------------|-------|--------|--------------|--------------|-------\n");

    int configs[][2] = {{1000, 1000}, {1000, 1000}, {1000, 1000}};
    float densities[] = {0.01f, 0.05f, 0.10f};
    int n = 3;

    for (int c = 0; c < n; c++)
        bench_config(configs[c][0], configs[c][1], densities[c]);
    return 0;
}