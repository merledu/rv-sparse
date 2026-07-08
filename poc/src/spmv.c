#include "rv_sparse.h"

rv_sparse_err_t rv_spmv_csr(const rv_csr_t* A, const float* x, float* y) {
    if (!A || !x || !y) return RV_SPARSE_ERR_ALLOC;
    
    for (uint32_t i = 0; i < A->M; i++) {
        float acc = 0.0f;
        for (uint32_t j = A->rowptr[i]; j < A->rowptr[i+1]; j++) {
            acc += A->val[j] * x[A->colidx[j]];
        }
        y[i] = acc;
    }
    
    return RV_SPARSE_SUCCESS;
}
