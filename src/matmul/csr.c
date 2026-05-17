#include "../../include/rv_sparse.h"

CSRMatrix_f32 *csr_alloc_f32(int rows, int cols, int nnz)
{
    CSRMatrix_f32 *A = malloc(sizeof(CSRMatrix_f32));
    A->num_rows = rows;
    A->num_cols = cols;
    A->nnz = nnz;
    A->values = malloc(nnz * sizeof(float));
    A->col_idx = malloc(nnz * sizeof(int));
    A->row_ptr = malloc((rows + 1) * sizeof(int));
    return A;
}

void csr_free_f32(CSRMatrix_f32 *A)
{
    free(A->values);
    free(A->col_idx);
    free(A->row_ptr);
    free(A);
}

CSRMatrix_f64 *csr_alloc_f64(int rows, int cols, int nnz)
{
    CSRMatrix_f64 *A = malloc(sizeof(CSRMatrix_f64));
    A->num_rows = rows;
    A->num_cols = cols;
    A->nnz = nnz;
    A->values = malloc(nnz * sizeof(double));
    A->col_idx = malloc(nnz * sizeof(int));
    A->row_ptr = malloc((rows + 1) * sizeof(int));
    return A;
}

void csr_free_f64(CSRMatrix_f64 *A)
{
    free(A->values);
    free(A->col_idx);
    free(A->row_ptr);
    free(A);
}
