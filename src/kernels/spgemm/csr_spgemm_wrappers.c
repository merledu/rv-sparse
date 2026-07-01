/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * Internal CSR SpGEMM wrapper layer.
 *
 * This file connects the public CSR matrix descriptors used by the API
 * with the raw pointer-based backend kernels.
 */

#include <stddef.h>
#include "rv_sparse.h"
#include "csr_spgemm_kernels.h"

// wrapper int8 kernel
rvsp_status_t rvsp_spgemm_csr_scalar_i8(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C)
{
    rvsp_status_t status;

    if (!A || !B || !C)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    status = rvsp_csr_validate(A);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    status = rvsp_csr_validate(B);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    if (A->dtype != RVSP_DTYPE_INT8 || B->dtype != RVSP_DTYPE_INT8)
    {
        return RVSP_ERROR_UNSUPPORTED_DTYPE;
    }

    if (A->cols != B->rows)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t *c_row_ptr = NULL;
    int32_t *c_col_idx = NULL;
    int32_t *c_values = NULL;
    int32_t c_nnz = 0;

    status = rvsp_spgemm_csr_scalar_i8_raw(
        A->rows,
        A->cols,
        B->cols,
        A->row_ptr,
        A->col_idx,
        (const int8_t *)A->values,
        B->row_ptr,
        B->col_idx,
        (const int8_t *)B->values,
        &c_row_ptr,
        &c_col_idx,
        &c_values,
        &c_nnz);

    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = c_nnz;
    C->row_ptr = c_row_ptr;
    C->col_idx = c_col_idx;
    C->values = c_values;
    C->dtype = RVSP_DTYPE_INT32;
    C->format = RVSP_FORMAT_CSR;
    C->owns_data = 1;

    return RVSP_SUCCESS;
}

// wrapper base scalar f32 kernel
rvsp_status_t rvsp_spgemm_csr_scalar_f32(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C)
{
    rvsp_status_t status;

    if (!A || !B || !C)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    status = rvsp_csr_validate(A);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    status = rvsp_csr_validate(B);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    if (A->dtype != RVSP_DTYPE_FP32 || B->dtype != RVSP_DTYPE_FP32)
    {
        return RVSP_ERROR_UNSUPPORTED_DTYPE;
    }

    if (A->cols != B->rows)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t *c_row_ptr = 0;
    int32_t *c_col_idx = 0;
    float *c_values = 0;
    int32_t c_nnz = 0;

    status = rvsp_spgemm_csr_scalar_f32_raw(
        A->rows,
        A->cols,
        B->cols,
        A->row_ptr,
        A->col_idx,
        (const float *)A->values,
        B->row_ptr,
        B->col_idx,
        (const float *)B->values,
        &c_row_ptr,
        &c_col_idx,
        &c_values,
        &c_nnz);

    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = c_nnz;
    C->row_ptr = c_row_ptr;
    C->col_idx = c_col_idx;
    C->values = c_values;
    C->dtype = RVSP_DTYPE_FP32;
    C->format = RVSP_FORMAT_CSR;
    C->owns_data = 1;

    return RVSP_SUCCESS;
}

// Wrapper for scalar_unroll4
rvsp_status_t rvsp_spgemm_csr_scalar_unroll4_f32(const rvsp_csr_matrix_t *A, const rvsp_csr_matrix_t *B,
                                                 rvsp_csr_matrix_t *C)
{
    rvsp_status_t status;

    if (!A || !B || !C)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    status = rvsp_csr_validate(A);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    status = rvsp_csr_validate(B);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    if (A->dtype != RVSP_DTYPE_FP32 || B->dtype != RVSP_DTYPE_FP32)
    {
        return RVSP_ERROR_UNSUPPORTED_DTYPE;
    }

    if (A->cols != B->rows)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t *c_row_ptr = NULL;
    int32_t *c_col_idx = NULL;
    float *c_values = NULL;
    int32_t c_nnz = 0;

    status = rvsp_spgemm_csr_scalar_unroll4_f32_raw(
        A->rows,
        A->cols,
        B->cols,
        A->row_ptr,
        A->col_idx,
        (const float *)A->values,
        B->row_ptr,
        B->col_idx,
        (const float *)B->values,
        &c_row_ptr,
        &c_col_idx,
        &c_values,
        &c_nnz);

    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = c_nnz;
    C->row_ptr = c_row_ptr;
    C->col_idx = c_col_idx;
    C->values = c_values;
    C->dtype = RVSP_DTYPE_FP32;
    C->format = RVSP_FORMAT_CSR;
    C->owns_data = 1;

    return RVSP_SUCCESS;
}

// wrapper for i8_indexed
rvsp_status_t rvsp_spgemm_csr_rvv_i8_indexed_marked(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C)
{
    if (A == NULL || B == NULL || C == NULL)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (A->dtype != RVSP_DTYPE_INT8 || B->dtype != RVSP_DTYPE_INT8)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (A->cols != B->rows)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t *c_row_ptr = NULL;
    int32_t *c_col_idx = NULL;
    int32_t *c_values = NULL;
    int32_t c_nnz = 0;

    rvsp_status_t status = rvsp_spgemm_csr_rvv_i8_indexed_marked_raw(
        A->rows,
        A->cols,
        B->cols,
        A->row_ptr,
        A->col_idx,
        (const int8_t *)A->values,
        B->row_ptr,
        B->col_idx,
        (const int8_t *)B->values,
        &c_row_ptr,
        &c_col_idx,
        &c_values,
        &c_nnz);

    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = c_nnz;
    C->row_ptr = c_row_ptr;
    C->col_idx = c_col_idx;
    C->values = c_values;
    C->dtype = RVSP_DTYPE_INT32;

    return RVSP_SUCCESS;
}

// wrapper for f32_indexed
rvsp_status_t rvsp_spgemm_csr_rvv_f32_indexed_marked(const rvsp_csr_matrix_t *A,
                                                     const rvsp_csr_matrix_t *B,
                                                     rvsp_csr_matrix_t *C)
{
    if (A == NULL || B == NULL || C == NULL)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (A->dtype != RVSP_DTYPE_FP32 || B->dtype != RVSP_DTYPE_FP32)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (A->cols != B->rows)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t *c_row_ptr = NULL;
    int32_t *c_col_idx = NULL;
    float *c_values = NULL;
    int32_t c_nnz = 0;

    rvsp_status_t status = rvsp_spgemm_csr_rvv_f32_indexed_marked_raw(
        A->rows,
        A->cols,
        B->cols,
        A->row_ptr,
        A->col_idx,
        (const float *)A->values,
        B->row_ptr,
        B->col_idx,
        (const float *)B->values,
        &c_row_ptr,
        &c_col_idx,
        &c_values,
        &c_nnz);

    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = c_nnz;
    C->row_ptr = c_row_ptr;
    C->col_idx = c_col_idx;
    C->values = c_values;
    C->dtype = RVSP_DTYPE_FP32;

    return RVSP_SUCCESS;
}