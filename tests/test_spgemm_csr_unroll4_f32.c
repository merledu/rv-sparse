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

static void assert_same_csr_f32(
    const rvsp_csr_matrix_t *expected,
    const rvsp_csr_matrix_t *actual)
{
    assert(expected->rows == actual->rows);
    assert(expected->cols == actual->cols);
    assert(expected->nnz == actual->nnz);

    for (int32_t i = 0; i < expected->rows + 1; i++)
    {
        assert(expected->row_ptr[i] == actual->row_ptr[i]);
    }

    for (int32_t i = 0; i < expected->nnz; i++)
    {
        assert(expected->col_idx[i] == actual->col_idx[i]);
    }

    const float *expected_values = (const float *)expected->values;
    const float *actual_values = (const float *)actual->values;

    for (int32_t i = 0; i < expected->nnz; i++)
    {
        assert(float_equal(expected_values[i], actual_values[i]));
    }
}

int main(void)
{
    /*
     * A =
     * [1 2 0 4]
     * [0 3 5 0]
     */
    int32_t a_row_ptr[] = {0, 3, 5};
    int32_t a_col_idx[] = {0, 1, 3, 1, 2};
    float a_values[] = {1.0f, 2.0f, 4.0f, 3.0f, 5.0f};

    /*
     * B =
     * [1 0 2]
     * [0 3 0]
     * [4 0 5]
     * [0 6 7]
     */
    int32_t b_row_ptr[] = {0, 2, 3, 5, 7};
    int32_t b_col_idx[] = {0, 2, 1, 0, 2, 1, 2};
    float b_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    rvsp_csr_matrix_t A;
    rvsp_csr_matrix_t B;
    rvsp_csr_matrix_t C_scalar = {0};
    rvsp_csr_matrix_t C_unroll4 = {0};

    rvsp_status_t status;

    status = rvsp_csr_create(
        &A,
        2,
        4,
        5,
        a_row_ptr,
        a_col_idx,
        a_values,
        RVSP_DTYPE_FP32);
    assert(status == RVSP_SUCCESS);

    status = rvsp_csr_create(
        &B,
        4,
        3,
        7,
        b_row_ptr,
        b_col_idx,
        b_values,
        RVSP_DTYPE_FP32);
    assert(status == RVSP_SUCCESS);

    rvsp_spgemm_options_t scalar_options = {
        .backend = RVSP_BACKEND_SCALAR,
        .input_dtype = RVSP_DTYPE_FP32,
        .output_dtype = RVSP_DTYPE_FP32,
        .sort_output_indices = 1};

    rvsp_spgemm_options_t unroll4_options = {
        .backend = RVSP_BACKEND_SCALAR_UNROLL4,
        .input_dtype = RVSP_DTYPE_FP32,
        .output_dtype = RVSP_DTYPE_FP32,
        .sort_output_indices = 1};

    status = rvsp_spgemm_csr(&A, &B, &C_scalar, &scalar_options);
    assert(status == RVSP_SUCCESS);

    status = rvsp_spgemm_csr(&A, &B, &C_unroll4, &unroll4_options);
    assert(status == RVSP_SUCCESS);

    assert_same_csr_f32(&C_scalar, &C_unroll4);

    rvsp_csr_destroy(&A);
    rvsp_csr_destroy(&B);
    rvsp_csr_destroy(&C_scalar);
    rvsp_csr_destroy(&C_unroll4);

    printf("test_spgemm_csr_unroll4_f32: PASS\n");

    return 0;
}