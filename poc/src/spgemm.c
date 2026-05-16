#include "rv_sparse.h"
#include <stdlib.h>
#include <string.h>

// Two-Phase Gustavson's Algorithm
// C = A * B
rv_sparse_err_t rv_spgemm_csr(const rv_csr_t* A, const rv_csr_t* B, rv_csr_t* C) {
    if (!A || !B || !C) return RV_SPARSE_ERR_ALLOC;
    if (A->N != B->M) return RV_SPARSE_ERR_MATH;

    uint32_t M = A->M;
    uint32_t N = B->N;

    // Allocate rowptr for C
    C->M = M;
    C->N = N;
    C->rowptr = (uint32_t*)calloc(M + 1, sizeof(uint32_t));
    if (!C->rowptr) return RV_SPARSE_ERR_ALLOC;

    // ----------------------------------------------------
    // PHASE 1: Symbolic (Compute exact nnz)
    // ----------------------------------------------------
    
    // marker array to track which columns have been seen in the current row
    // Initialize to -1
    int32_t* marker = (int32_t*)malloc(N * sizeof(int32_t));
    for (uint32_t i = 0; i < N; i++) marker[i] = -1;

    uint32_t total_nnz = 0;

    for (uint32_t i = 0; i < M; i++) {
        uint32_t row_nnz = 0;
        
        // Loop over non-zeros of A in row i
        for (uint32_t j = A->rowptr[i]; j < A->rowptr[i+1]; j++) {
            uint32_t a_col = A->colidx[j];
            
            // Loop over non-zeros of B in row a_col
            for (uint32_t k = B->rowptr[a_col]; k < B->rowptr[a_col+1]; k++) {
                uint32_t b_col = B->colidx[k];
                
                // If we haven't seen this column in the current row yet
                if (marker[b_col] != (int32_t)i) {
                    marker[b_col] = i;
                    row_nnz++;
                }
            }
        }
        total_nnz += row_nnz;
        C->rowptr[i+1] = total_nnz;
    }
    
    C->nnz = total_nnz;

    // Allocate colidx and val for C exactly once
    if (total_nnz > 0) {
        C->colidx = (uint32_t*)malloc(total_nnz * sizeof(uint32_t));
        C->val = (float*)malloc(total_nnz * sizeof(float));
        if (!C->colidx || !C->val) {
            free(C->rowptr);
            free(marker);
            return RV_SPARSE_ERR_ALLOC;
        }
    } else {
        C->colidx = NULL;
        C->val = NULL;
        free(marker);
        return RV_SPARSE_SUCCESS;
    }

    // ----------------------------------------------------
    // PHASE 2: Numeric (Compute values safely)
    // ----------------------------------------------------
    
    // We reuse marker array, reset it to -1
    for (uint32_t i = 0; i < N; i++) marker[i] = -1;
    
    // Workspace to accumulate floating point values for a single row
    // This is allocated ONCE outside the loop, solving the calloc bottleneck
    float* workspace = (float*)calloc(N, sizeof(float));
    
    // Array to track which indices in workspace were touched
    // This solves the O(K) dense scan bottleneck
    uint32_t* touched = (uint32_t*)malloc(N * sizeof(uint32_t));

    uint32_t c_idx = 0;

    for (uint32_t i = 0; i < M; i++) {
        uint32_t touched_count = 0;
        
        // Accumulate values
        for (uint32_t j = A->rowptr[i]; j < A->rowptr[i+1]; j++) {
            uint32_t a_col = A->colidx[j];
            float a_val = A->val[j];
            
            for (uint32_t k = B->rowptr[a_col]; k < B->rowptr[a_col+1]; k++) {
                uint32_t b_col = B->colidx[k];
                float b_val = B->val[k];
                
                workspace[b_col] += a_val * b_val;
                
                if (marker[b_col] != (int32_t)i) {
                    marker[b_col] = i;
                    touched[touched_count++] = b_col;
                }
            }
        }
        
        // Gather values into C
        // Only loop over touched elements instead of full N
        for (uint32_t t = 0; t < touched_count; t++) {
            uint32_t col = touched[t];
            C->colidx[c_idx] = col;
            C->val[c_idx] = workspace[col];
            c_idx++;
            
            // Reset workspace explicitly for next iteration
            workspace[col] = 0.0f;
        }
    }

    free(marker);
    free(workspace);
    free(touched);

    return RV_SPARSE_SUCCESS;
}
