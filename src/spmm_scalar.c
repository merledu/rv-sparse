#include "rv_sparse.h"

void rvs_csr_spmm_scalar(const rvs_csr_t *A, const rvs_float_t *B, rvs_float_t *C, rvs_index_t B_cols) {
    // Basic scalar SpMM (Dense B and C, Sparse A)
    for (rvs_index_t i = 0; i < A->rows; ++i) {
        for (rvs_index_t j = 0; j < B_cols; ++j) {
            rvs_float_t sum = 0.0f;
            for (rvs_index_t k = A->row_ptr[i]; k < A->row_ptr[i+1]; ++k) {
                sum += A->values[k] * B[A->col_indices[k] * B_cols + j];
            }
            C[i * B_cols + j] = sum;
        }
    }
}
