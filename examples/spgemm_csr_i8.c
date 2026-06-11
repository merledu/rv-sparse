#include <stdint.h>
#include "rv_sparse.h"

int main(void)
{
    int32_t a_row_ptr[] = {0, 2, 3};
    int32_t a_col_idx[] = {0, 1, 1};
    int8_t a_values[] = {1, 2, 3};

    int32_t b_row_ptr[] = {0, 1, 2};
    int32_t b_col_idx[] = {0, 1};
    int8_t b_values[] = {4, 5};

    rvsp_csr_matrix_t A;
    rvsp_csr_matrix_t B;
    rvsp_csr_matrix_t C;

    rvsp_csr_create(
        &A,
        2,
        2,
        3,
        a_row_ptr,
        a_col_idx,
        a_values,
        RVSP_DTYPE_INT8);

    rvsp_csr_create(
        &B,
        2,
        2,
        2,
        b_row_ptr,
        b_col_idx,
        b_values,
        RVSP_DTYPE_INT8);

    rvsp_spgemm_options_t options = {
        .backend = RVSP_BACKEND_SCALAR,
        .input_dtype = RVSP_DTYPE_INT8,
        .output_dtype = RVSP_DTYPE_INT32,
        .sort_output_indices = 1};

    rvsp_spgemm_csr(&A, &B, &C, &options);

    rvsp_csr_destroy(&A);
    rvsp_csr_destroy(&B);
    rvsp_csr_destroy(&C);

    return 0;
}
