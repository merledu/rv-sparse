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
 * This file implements a CSR SpGEMM backend kernel. The kernel is intended
 * to be called through the internal wrapper and dispatcher layers, not
 * directly by users of the public API.
 */

#include <stdlib.h>
#include <stdlib.h>
#include "rv_sparse.h"
#include "csr_spgemm_kernels.h"

rvsp_status_t rvsp_spgemm_csr_scalar_f32_raw(int32_t a_rows, int32_t a_cols, int32_t b_cols,
                                             const int32_t *restrict a_row_ptr, const int32_t *restrict a_col_idx,
                                             const float *restrict a_values, const int32_t *restrict b_row_ptr, const int32_t *restrict b_col_idx,
                                             const float *restrict b_values, int32_t **c_row_ptr_out, int32_t **c_col_idx_out,
                                             float **c_values_out, int32_t *c_nnz_out)
{
    if (!a_row_ptr || !a_col_idx || !a_values ||
        !b_row_ptr || !b_col_idx || !b_values ||
        !c_row_ptr_out || !c_col_idx_out || !c_values_out || !c_nnz_out)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (a_rows <= 0 || a_cols <= 0 || b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t max_nnz = a_rows * b_cols;

    int32_t *c_row_ptr = (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));
    int32_t *c_col_idx = (int32_t *)malloc((size_t)max_nnz * sizeof(int32_t));
    float *c_values = (float *)malloc((size_t)max_nnz * sizeof(float));

    if (!c_row_ptr || !c_col_idx || !c_values)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    int32_t nnz_count = 0;

    for (int32_t row = 0; row < a_rows; row++)
    {
        c_row_ptr[row] = nnz_count;

        float *acc = (float *)calloc((size_t)b_cols, sizeof(float));

        if (!acc)
        {
            free(c_row_ptr);
            free(c_col_idx);
            free(c_values);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }

        int32_t a_start = a_row_ptr[row];
        int32_t a_end = a_row_ptr[row + 1];

        for (int32_t a_pos = a_start; a_pos < a_end; a_pos++)
        {
            int32_t k = a_col_idx[a_pos];
            float a_val = a_values[a_pos];

            int32_t b_start = b_row_ptr[k];
            int32_t b_end = b_row_ptr[k + 1];

            for (int32_t b_pos = b_start; b_pos < b_end; b_pos++)
            {
                int32_t col = b_col_idx[b_pos];
                float b_val = b_values[b_pos];

                acc[col] += a_val * b_val;
            }
        }

        for (int32_t col = 0; col < b_cols; col++)
        {
            if (acc[col] != 0.0f)
            {
                c_col_idx[nnz_count] = col;
                c_values[nnz_count] = acc[col];
                nnz_count++;
            }
        }

        free(acc);
    }

    c_row_ptr[a_rows] = nnz_count;

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = nnz_count;

    (void)a_cols;

    return RVSP_SUCCESS;
}