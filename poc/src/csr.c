#include "rv_sparse.h"
#include <stdlib.h>

rv_sparse_err_t rv_csr_create(rv_csr_t* mat, uint32_t M, uint32_t N, uint32_t nnz) {
    if (!mat) return RV_SPARSE_ERR_ALLOC;
    
    mat->M = M;
    mat->N = N;
    mat->nnz = nnz;
    
    mat->rowptr = (uint32_t*)calloc(M + 1, sizeof(uint32_t));
    if (!mat->rowptr) return RV_SPARSE_ERR_ALLOC;
    
    if (nnz > 0) {
        mat->colidx = (uint32_t*)malloc(nnz * sizeof(uint32_t));
        mat->val = (float*)malloc(nnz * sizeof(float));
        
        if (!mat->colidx || !mat->val) {
            free(mat->rowptr);
            free(mat->colidx);
            free(mat->val);
            return RV_SPARSE_ERR_ALLOC;
        }
    } else {
        mat->colidx = NULL;
        mat->val = NULL;
    }
    
    return RV_SPARSE_SUCCESS;
}

void rv_csr_destroy(rv_csr_t* mat) {
    if (mat) {
        free(mat->rowptr);
        free(mat->colidx);
        free(mat->val);
        mat->rowptr = NULL;
        mat->colidx = NULL;
        mat->val = NULL;
        mat->M = 0;
        mat->N = 0;
        mat->nnz = 0;
    }
}
