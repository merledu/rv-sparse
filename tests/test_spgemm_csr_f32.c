/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Correctness test for the rv-sparse public API.
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "rv_sparse.h"

static int float_equal(float a, float b)
{
    return fabsf(a - b) < 1e-5f;
}

int main(void)
{
    /*
     * A =
     * [1 2]
     * [0 3]
     */
    int32_t a_row_ptr[] = {0, 2, 3};
    int32_t a_col_idx[] = {0, 1, 1};
    float a_values[] = {1.0f, 2.0f, 3.0f};

    /*
     * B =
     * [4 0]
     * [0 5]
     */
    int32_t b_row_ptr[] = {0, 1, 2};
    int32_t b_col_idx[] = {0, 1};
    float b_values[] = {4.0f, 5.0f};

    rvsp_csr_matrix_t A;
    rvsp_csr_matrix_t B;
    rvsp_csr_matrix_t C = {0};

    rvsp_status_t status;

    status = rvsp_csr_create(
        &A,
        2,
        2,
        3,
        a_row_ptr,
        a_col_idx,
        a_values,
        RVSP_DTYPE_FP32);
    assert(status == RVSP_SUCCESS);

    status = rvsp_csr_create(
        &B,
        2,
        2,
        2,
        b_row_ptr,
        b_col_idx,
        b_values,
        RVSP_DTYPE_FP32);
    assert(status == RVSP_SUCCESS);

    rvsp_spgemm_options_t options = {
        .backend = RVSP_BACKEND_SCALAR,
        .input_dtype = RVSP_DTYPE_FP32,
        .output_dtype = RVSP_DTYPE_FP32,
        .sort_output_indices = 1};

    status = rvsp_spgemm_csr(&A, &B, &C, &options);
    assert(status == RVSP_SUCCESS);

    assert(C.rows == 2);
    assert(C.cols == 2);
    assert(C.nnz == 3);

    assert(C.row_ptr[0] == 0);
    assert(C.row_ptr[1] == 2);
    assert(C.row_ptr[2] == 3);

    assert(C.col_idx[0] == 0);
    assert(C.col_idx[1] == 1);
    assert(C.col_idx[2] == 1);

    float *c_values = (float *)C.values;

    assert(float_equal(c_values[0], 4.0f));
    assert(float_equal(c_values[1], 10.0f));
    assert(float_equal(c_values[2], 15.0f));

    rvsp_csr_destroy(&A);
    rvsp_csr_destroy(&B);
    rvsp_csr_destroy(&C);

    printf("test_spgemm_csr_f32: PASS\n");

    return 0;
}