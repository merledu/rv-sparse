// src/spmv_rvv.c
#include "rv_sparse.h"
#include <riscv_vector.h>
#include <stdint.h>

#if defined(__GNUC__)
#define RVSPARSE_NOINLINE __attribute__((noinline))
#else
#define RVSPARSE_NOINLINE
#endif

RVSPARSE_NOINLINE void spmv_csr_rvv_f32(const CSRMatrix_f32 *A, const float *x, float *y)
{
    for (int row = 0; row < A->num_rows; row++)
    {
        int start = A->row_ptr[row];
        int end = A->row_ptr[row + 1];
        int row_len = end - start;

        size_t vlmax = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax); // Initialize accumulator to zero

        int i = 0;
        while (i < row_len)
        {
            size_t vl = __riscv_vsetvl_e32m1(row_len - i);
            vfloat32m1_t vvals = __riscv_vle32_v_f32m1(A->values + start + i, vl);
            vuint32m1_t vcols = __riscv_vle32_v_u32m1((const uint32_t *)(A->col_idx + start + i), vl);
            vuint32m1_t voffsets = __riscv_vsll_vx_u32m1(vcols, 2, vl);
            vfloat32m1_t vx = __riscv_vloxei32_v_f32m1(x, voffsets, vl);
            vacc = __riscv_vfmacc_vv_f32m1(vacc, vvals, vx, vl);
            i += vl;
        }
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        vfloat32m1_t vreduced = __riscv_vfredosum_vs_f32m1_f32m1(vacc, vzero, vlmax);
        y[row] = __riscv_vfmv_f_s_f32m1_f32(vreduced);
    }
}

RVSPARSE_NOINLINE void spmv_csr_rvv_f32_counted(const CSRMatrix_f32 *A,
                                                const float *x, float *y,
                                                uint64_t *ops, uint64_t *elems)
{
    for (int row = 0; row < A->num_rows; row++)
    {
        int start = A->row_ptr[row];
        int row_len = A->row_ptr[row + 1] - start;
        if (row_len == 0)
        {
            y[row] = 0.0f;
            continue;
        }

        size_t vlmax = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

        int i = 0;
        while (i < row_len)
        {
            size_t vl = __riscv_vsetvl_e32m1(row_len - i);

            vfloat32m1_t vvals = __riscv_vle32_v_f32m1(A->values + start + i, vl);
            vuint32m1_t vcols = __riscv_vle32_v_u32m1((const uint32_t *)(A->col_idx + start + i), vl);
            vuint32m1_t voffsets = __riscv_vsll_vx_u32m1(vcols, 2, vl);
            vfloat32m1_t vx = __riscv_vloxei32_v_f32m1(x, voffsets, vl);

            vacc = __riscv_vfmacc_vv_f32m1(vacc, vvals, vx, vl);

            (*ops)++;
            (*elems) += vl;
            i += vl;
        }

        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        vfloat32m1_t vsum = __riscv_vfredosum_vs_f32m1_f32m1(vacc, vzero, vlmax);
        y[row] = __riscv_vfmv_f_s_f32m1_f32(vsum);
    }
}

RVSPARSE_NOINLINE void spmv_csr_rvv_f64(const CSRMatrix_f64 *A, const double *x, double *y)
{
    for (int row = 0; row < A->num_rows; row++)
    {
        int start = A->row_ptr[row];
        int row_len = A->row_ptr[row + 1] - start;

        if (row_len == 0)
        {
            y[row] = 0.0;
            continue;
        }

        size_t vlmax = __riscv_vsetvlmax_e64m2();
        vfloat64m2_t vacc = __riscv_vfmv_v_f_f64m2(0.0, vlmax);

        int i = 0;
        while (i < row_len)
        {
            size_t vl = __riscv_vsetvl_e64m2(row_len - i);

            vfloat64m2_t vvals = __riscv_vle64_v_f64m2(
                A->values + start + i, vl);

            vuint32m1_t vcols32 = __riscv_vle32_v_u32m1(
                (const uint32_t *)(A->col_idx + start + i), vl);
            vuint64m2_t vcols64 = __riscv_vzext_vf2_u64m2(vcols32, vl);

            // shift left by 3 = multiply by 8 (sizeof double)
            vuint64m2_t voffsets = __riscv_vsll_vx_u64m2(vcols64, 3, vl);

            vfloat64m2_t vx = __riscv_vloxei64_v_f64m2(x, voffsets, vl);

            vacc = __riscv_vfmacc_vv_f64m2(vacc, vvals, vx, vl);

            i += vl;
        }

        vfloat64m1_t vzero = __riscv_vfmv_v_f_f64m1(0.0, 1);
        vfloat64m1_t vsum = __riscv_vfredosum_vs_f64m2_f64m1(vacc, vzero, vlmax);
        y[row] = __riscv_vfmv_f_s_f64m1_f64(vsum);
    }
}