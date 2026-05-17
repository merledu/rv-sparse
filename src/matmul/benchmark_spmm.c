// bench/benchmark_spmm.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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

typedef void (*spmm_fn)(const CSRMatrix_f32 *A, const float *B, int B_cols, float *C);

static uint64_t time_cycles_spmm(spmm_fn fn, const CSRMatrix_f32 *A,
                                 const float *B, int B_cols, float *C)
{
    for (int i = 0; i < WARMUP; i++)
        fn(A, B, B_cols, C);

    uint64_t t0 = rdcycle();
    for (int i = 0; i < RUNS; i++)
        fn(A, B, B_cols, C);

    return (rdcycle() - t0) / RUNS;
}

static void bench_spmm_config(int rows, int cols, float density, int b_cols)
{
    CSRMatrix_f32 *A = generate_random_csr_f32(rows, cols, density, 42);
    float *B = malloc((size_t)cols * b_cols * sizeof(float));
    float *C = malloc((size_t)rows * b_cols * sizeof(float));
    for (int i = 0; i < cols * b_cols; i++)
        B[i] = (float)i * 0.001f;

    uint64_t scalar_cycles = time_cycles_spmm(spmm_csr_scalar_f32, A, B, b_cols, C);

    uint64_t rvv_cycles = time_cycles_spmm(spmm_csr_rvv_f32, A, B, b_cols, C);
    double speedup = (rvv_cycles == 0) ? 0.0 : (double)scalar_cycles / rvv_cycles;

    printf("%5dx%-5d | %4.0f%% | %5d | %12lu | %12lu | %5.2fx\n",
           rows, cols, density * 100, b_cols,
           scalar_cycles, rvv_cycles, speedup);

    // free(B);
    // free(C);
    // csr_free_f32(A);
}

int main(void)
{
    printf("%-12s | %5s | %5s | %12s | %12s | %6s\n",
           "SpMM", "Dense", "Bcol", "Scalar cyc", "RVV cyc", "Speedup");
    printf("-------------|-------|-------|--------------|--------------|-------\n");

    bench_spmm_config(256, 256, 0.01f, 32);
    bench_spmm_config(512, 512, 0.01f, 32);
    bench_spmm_config(512, 512, 0.01f, 64);
    return 0;
}
