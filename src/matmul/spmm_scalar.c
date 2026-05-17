#include "../../include/rv_sparse.h"

void spmm_csr_scalar_f32(const CSRMatrix_f32 *A, const float *B, int B_cols, float *C)
{
    int M = A->num_rows;
    for (int r = 0; r < M; r++)
    {
        float *Crow = C + (size_t)r * B_cols;
        for (int j = 0; j < B_cols; j++)
            Crow[j] = 0.0f;

        for (int i = A->row_ptr[r]; i < A->row_ptr[r + 1]; i++)
        {
            int k = A->col_idx[i];
            float aval = A->values[i];
            const float *Brow = B + (size_t)k * B_cols;
            for (int j = 0; j < B_cols; j++)
                Crow[j] += aval * Brow[j];
        }
    }
}

void spmm_csr_scalar_f64(const CSRMatrix_f64 *A, const double *B, int B_cols, double *C)
{
    int M = A->num_rows;
    for (int r = 0; r < M; r++)
    {
        double *Crow = C + (size_t)r * B_cols;
        for (int j = 0; j < B_cols; j++)
            Crow[j] = 0.0;

        for (int i = A->row_ptr[r]; i < A->row_ptr[r + 1]; i++)
        {
            int k = A->col_idx[i];
            double aval = A->values[i];
            const double *Brow = B + (size_t)k * B_cols;
            for (int j = 0; j < B_cols; j++)
                Crow[j] += aval * Brow[j];
        }
    }
}
