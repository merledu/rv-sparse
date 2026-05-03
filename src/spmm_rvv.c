#include "rv_sparse.h"
#include <riscv_vector.h>

void rvs_csr_spmm_rvv(const rvs_csr_t *A, const rvs_float_t *B, rvs_float_t *C, rvs_index_t B_cols) {
    // Vectorize over the columns of dense matrices B and C.
    // This avoids expensive gather/scatter instructions and utilizes contiguous memory loads.
    
    // Zero out the C matrix
    for (rvs_index_t i = 0; i < A->rows * B_cols; ++i) {
        C[i] = 0.0f;
    }

    for (rvs_index_t i = 0; i < A->rows; ++i) {
        rvs_index_t row_start = A->row_ptr[i];
        rvs_index_t row_end = A->row_ptr[i+1];
        
        // Loop over the non-zero elements in row i of sparse matrix A
        for (rvs_index_t k = row_start; k < row_end; ++k) {
            rvs_float_t a_val = A->values[k];
            rvs_index_t b_row = A->col_indices[k];
            
            const rvs_float_t *B_ptr = &B[b_row * B_cols];
            rvs_float_t *C_ptr = &C[i * B_cols];
            
            rvs_index_t vl;
            // Inner loop: vectorized over the columns of dense matrix B
            for (rvs_index_t j = 0; j < B_cols; j += vl) {
                // Determine vector length for this strip
                vl = __riscv_vsetvl_e32m1(B_cols - j);
                
                // Contiguous load C[i, j...j+vl-1]
                vfloat32m1_t vec_C = __riscv_vle32_v_f32m1(C_ptr + j, vl);
                
                // Contiguous load B[b_row, j...j+vl-1]
                vfloat32m1_t vec_B = __riscv_vle32_v_f32m1(B_ptr + j, vl);
                
                // Multiply-accumulate: vec_C += a_val * vec_B
                vec_C = __riscv_vfmacc_vf_f32m1(vec_C, a_val, vec_B, vl);
                
                // Store back to C
                __riscv_vse32_v_f32m1(C_ptr + j, vec_C, vl);
            }
        }
    }
}
