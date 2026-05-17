#include "../../include/rv_sparse.h"

#if defined(__GNUC__)
#define RVSPARSE_NOINLINE __attribute__((noinline))
#else
#define RVSPARSE_NOINLINE
#endif

RVSPARSE_NOINLINE void spmv_csr_scalar_f32(const CSRMatrix_f32 *A, const float *x, float *y)
{
    for (int row = 0; row < A->num_rows; row++)
    {
        float sum = 0.0f;
        for (int i = A->row_ptr[row]; i < A->row_ptr[row + 1]; i++)
            sum += A->values[i] * x[A->col_idx[i]];
        y[row] = sum;
    }
}

RVSPARSE_NOINLINE void spmv_csr_scalar_f32_counted(const CSRMatrix_f32 *A,
                                                   const float *x, float *y,
                                                   uint64_t *ops)
{
    for (int row = 0; row < A->num_rows; row++)
    {
        float sum = 0.0f;
        for (int i = A->row_ptr[row]; i < A->row_ptr[row + 1]; i++)
        {
            sum += A->values[i] * x[A->col_idx[i]];
            (*ops)++;
        }
        y[row] = sum;
    }
}

RVSPARSE_NOINLINE void spmv_csr_scalar_f64(const CSRMatrix_f64 *A, const double *x, double *y)
{
    for (int row = 0; row < A->num_rows; row++)
    {
        double sum = 0.0;
        for (int i = A->row_ptr[row]; i < A->row_ptr[row + 1]; i++)
            sum += A->values[i] * x[A->col_idx[i]];
        y[row] = sum;
    }
}
