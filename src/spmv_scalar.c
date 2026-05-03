#include "rv_sparse.h"

void rvs_csr_spmv_scalar(const rvs_csr_t *A, const rvs_float_t *x, rvs_float_t *y) {
    for (rvs_index_t i = 0; i < A->rows; ++i) {
        rvs_float_t sum = 0.0f;
        for (rvs_index_t j = A->row_ptr[i]; j < A->row_ptr[i+1]; ++j) {
            sum += A->values[j] * x[A->col_indices[j]];
        }
        y[i] = sum;
    }
}
