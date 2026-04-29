#include "rv_sparse.h"
#include <riscv_vector.h>

void rvs_csr_spmv_rvv(const rvs_csr_t *A, const rvs_float_t *x, rvs_float_t *y) {
    for (rvs_index_t i = 0; i < A->rows; ++i) {
        rvs_index_t row_start = A->row_ptr[i];
        rvs_index_t row_end = A->row_ptr[i+1];
        rvs_index_t vl;
        
        // We use LMUL=1 (m1). You could scale to m2, m4 for longer vectors.
        size_t vlmax = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        
        for (rvs_index_t j = row_start; j < row_end; j += vl) {
            // Strip mining: Request vector length for remaining elements
            vl = __riscv_vsetvl_e32m1(row_end - j);
            
            // Contiguous load for A values
            vfloat32m1_t vec_A = __riscv_vle32_v_f32m1(&A->values[j], vl);
            
            // Contiguous load for column indices (cast to unsigned since vloxei32 expects vuint32m1_t)
            vuint32m1_t vec_col_idx = __riscv_vle32_v_u32m1((const uint32_t*)&A->col_indices[j], vl);
            
            // Indexed load (gather) from the dense vector 'x' using column indices.
            vfloat32m1_t vec_x = __riscv_vloxei32_v_f32m1(x, vec_col_idx, vl);
            
            // Multiply and accumulate: vec_sum += vec_A * vec_x
            vec_sum = __riscv_vfmacc_vv_f32m1(vec_sum, vec_A, vec_x, vl);
        }
        
        // Sum reduction across the vector register
        // vfredusum (unordered) is faster than vfredosum (ordered)
        vfloat32m1_t vec_red_init = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        vfloat32m1_t vec_red = __riscv_vfredusum_vs_f32m1_f32m1(vec_sum, vec_red_init, vlmax);
        
        // Extract the scalar result and store it
        y[i] = __riscv_vfmv_f_s_f32m1_f32(vec_red);
    }
}
