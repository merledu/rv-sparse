#include "rv_sparse.h"
#include <riscv_vector.h>

void spmm_csr_rvv_f32(const CSRMatrix_f32 *A, const float *B, int B_cols, float *C)
{
    int M = A->num_rows;
    for (int r = 0; r < M; r++)
    {
        float *Crow = C + (size_t)r * B_cols;
        int start = A->row_ptr[r];
        int end = A->row_ptr[r + 1];

        for (int col = 0; col < B_cols; col += (int)__riscv_vsetvlmax_e32m1())
        {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(B_cols - col));
            vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (int i = start; i < end; i++)
            {
                int k = A->col_idx[i];
                const float *Brow = B + (size_t)k * B_cols + col;
                vfloat32m1_t vb = __riscv_vle32_v_f32m1(Brow, vl);
                vacc = __riscv_vfmacc_vf_f32m1(vacc, A->values[i], vb, vl);
            }

            __riscv_vse32_v_f32m1(Crow + col, vacc, vl);
        }
    }
}

void spmm_csr_rvv_f64(const CSRMatrix_f64 *A, const double *B, int B_cols, double *C)
{
    int M = A->num_rows;
    for (int r = 0; r < M; r++)
    {
        double *Crow = C + (size_t)r * B_cols;
        int start = A->row_ptr[r];
        int end = A->row_ptr[r + 1];

        for (int col = 0; col < B_cols; col += (int)__riscv_vsetvlmax_e64m2())
        {
            size_t vl = __riscv_vsetvl_e64m2((size_t)(B_cols - col));
            vfloat64m2_t vacc = __riscv_vfmv_v_f_f64m2(0.0, vl);

            for (int i = start; i < end; i++)
            {
                int k = A->col_idx[i];
                const double *Brow = B + (size_t)k * B_cols + col;
                vfloat64m2_t vb = __riscv_vle64_v_f64m2(Brow, vl);
                vacc = __riscv_vfmacc_vf_f64m2(vacc, A->values[i], vb, vl);
            }

            __riscv_vse64_v_f64m2(Crow + col, vacc, vl);
        }
    }
}
