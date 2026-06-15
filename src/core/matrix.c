/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * rv-sparse is an experimental sparse linear algebra library focused on
 * portable sparse kernels and future RISC-V optimization.
 *
 * This source file implements part of the internal backend infrastructure
 * used by the public API.
 */

#include <stdlib.h>
#include "rv_sparse.h"

rvsp_status_t rvsp_csr_create(
    rvsp_csr_matrix_t *A,
    int32_t rows,
    int32_t cols,
    int32_t nnz,
    int32_t *row_ptr,
    int32_t *col_idx,
    void *values,
    rvsp_dtype_t dtype)
{
    if (!A || !row_ptr || !col_idx || !values)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (rows <= 0 || cols <= 0 || nnz < 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    A->rows = rows;
    A->cols = cols;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_idx = col_idx;
    A->values = values;
    A->dtype = dtype;
    A->format = RVSP_FORMAT_CSR;
    A->owns_data = 0;

    return rvsp_csr_validate(A);
}

rvsp_status_t rvsp_csr_validate(const rvsp_csr_matrix_t *A)
{
    if (!A || !A->row_ptr || !A->col_idx || !A->values)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (A->rows <= 0 || A->cols <= 0 || A->nnz < 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (A->row_ptr[0] != 0)
    {
        return RVSP_ERROR_INVALID_CSR;
    }

    if (A->row_ptr[A->rows] != A->nnz)
    {
        return RVSP_ERROR_INVALID_CSR;
    }

    for (int32_t i = 0; i < A->rows; i++)
    {
        if (A->row_ptr[i] > A->row_ptr[i + 1])
        {
            return RVSP_ERROR_INVALID_CSR;
        }
    }

    for (int32_t i = 0; i < A->nnz; i++)
    {
        if (A->col_idx[i] < 0 || A->col_idx[i] >= A->cols)
        {
            return RVSP_ERROR_INVALID_CSR;
        }
    }

    return RVSP_SUCCESS;
}

void rvsp_csr_destroy(rvsp_csr_matrix_t *A)
{
    if (!A)
    {
        return;
    }

    if (A->owns_data)
    {
        free(A->row_ptr);
        free(A->col_idx);
        free(A->values);
    }

    A->rows = 0;
    A->cols = 0;
    A->nnz = 0;
    A->row_ptr = NULL;
    A->col_idx = NULL;
    A->values = NULL;
    A->dtype = RVSP_DTYPE_FP32;
    A->format = RVSP_FORMAT_CSR;
    A->owns_data = 0;
}
