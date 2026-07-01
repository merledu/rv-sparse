/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Experimental optimized CSR SpGEMM RVV INT8 raw kernel.
 *
 * Computes INT8 x INT8 with INT32 accumulation/output.
 *
 * Optimization strategy:
 * - Reuse one dense accumulator across rows.
 * - Track touched columns using marker/touched arrays.
 * - Avoid scanning all B columns when constructing each output row.
 * - Use RVV indexed gather/scatter for row accumulation.
 */

#include <stdint.h>
#include <stdlib.h>

#include "csr_spgemm_kernels.h"

static int compare_i32(const void *a, const void *b)
{
    int32_t x = *(const int32_t *)a;
    int32_t y = *(const int32_t *)b;

    return (x > y) - (x < y);
}

static rvsp_status_t reserve_output_capacity(int32_t **col_idx, int32_t **values,
                                             int32_t *capacity,
                                             int32_t required)
{
    if (required <= *capacity)
    {
        return RVSP_SUCCESS;
    }

    int32_t new_capacity = *capacity > 0 ? *capacity : 64;

    while (new_capacity < required)
    {
        if (new_capacity > INT32_MAX / 2)
        {
            return RVSP_ERROR_ALLOCATION_FAILED;
        }

        new_capacity *= 2;
    }

    int32_t *new_col_idx = (int32_t *)realloc(
        *col_idx,
        (size_t)new_capacity * sizeof(int32_t));

    if (!new_col_idx)
    {
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    int32_t *new_values = (int32_t *)realloc(
        *values,
        (size_t)new_capacity * sizeof(int32_t));

    if (!new_values)
    {
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    *col_idx = new_col_idx;
    *values = new_values;
    *capacity = new_capacity;

    return RVSP_SUCCESS;
}

rvsp_status_t rvsp_spgemm_csr_rvv_i8_indexed_marked_raw(int32_t a_rows, int32_t a_cols, int32_t b_cols,
                                                       const int32_t *a_row_ptr, const int32_t *a_col_idx,
                                                       const int8_t *a_values, const int32_t *b_row_ptr,
                                                       const int32_t *b_col_idx, const int8_t *b_values,
                                                       int32_t **c_row_ptr_out, int32_t **c_col_idx_out,
                                                       int32_t **c_values_out, int32_t *c_nnz_out)
{
    if (!a_row_ptr || !a_col_idx || !a_values ||
        !b_row_ptr || !b_col_idx || !b_values ||
        !c_row_ptr_out || !c_col_idx_out ||
        !c_values_out || !c_nnz_out)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (a_rows <= 0 || a_cols <= 0 || b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t *c_row_ptr = (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));

    int32_t output_capacity = 64;
    int32_t *c_col_idx = (int32_t *)malloc((size_t)output_capacity * sizeof(int32_t));
    int32_t *c_values = (int32_t *)malloc((size_t)output_capacity * sizeof(int32_t));

    int32_t *acc = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));
    int32_t *marker = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));
    int32_t *touched = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));

    if (!c_row_ptr || !c_col_idx || !c_values || !acc || !marker || !touched)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        free(acc);
        free(marker);
        free(touched);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    for (int32_t col = 0; col < b_cols; col++)
    {
        marker[col] = -1;
        acc[col] = 0;
    }

    int32_t nnz_count = 0;

    for (int32_t row = 0; row < a_rows; row++)
    {
        c_row_ptr[row] = nnz_count;

        int32_t touched_count = 0;

        int32_t a_start = a_row_ptr[row];
        int32_t a_end = a_row_ptr[row + 1];

        for (int32_t a_pos = a_start; a_pos < a_end; a_pos++)
        {
            int32_t k = a_col_idx[a_pos];

            if (k < 0 || k >= a_cols)
            {
                free(c_row_ptr);
                free(c_col_idx);
                free(c_values);
                free(acc);
                free(marker);
                free(touched);
                return RVSP_ERROR_INVALID_CSR;
            }

            int32_t b_start = b_row_ptr[k];
            int32_t b_end = b_row_ptr[k + 1];

            /*
             * Prepass:
             * - validate columns
             * - register touched columns
             * - initialize accumulator entries for the current row
             */
            for (int32_t b_pos = b_start; b_pos < b_end; b_pos++)
            {
                int32_t col = b_col_idx[b_pos];

                if (col < 0 || col >= b_cols)
                {
                    free(c_row_ptr);
                    free(c_col_idx);
                    free(c_values);
                    free(acc);
                    free(marker);
                    free(touched);
                    return RVSP_ERROR_INVALID_CSR;
                }

                if (marker[col] != row)
                {
                    marker[col] = row;
                    touched[touched_count] = col;
                    touched_count++;
                    acc[col] = 0;
                }
            }

            rvsp_status_t status = rvsp_accumulate_row_i8_rvv_indexed_fast(a_values[a_pos], b_end - b_start,
                                                                          &b_col_idx[b_start], &b_values[b_start],
                                                                          acc);

            if (status != RVSP_SUCCESS)
            {
                free(c_row_ptr);
                free(c_col_idx);
                free(c_values);
                free(acc);
                free(marker);
                free(touched);
                return status;
            }
        }

        // Keep output deterministic and comparable with the scalar reference.
        qsort(touched, (size_t)touched_count, sizeof(int32_t), compare_i32);

        rvsp_status_t status = reserve_output_capacity(&c_col_idx, &c_values,
                                                       &output_capacity, nnz_count + touched_count);

        if (status != RVSP_SUCCESS)
        {
            free(c_row_ptr);
            free(c_col_idx);
            free(c_values);
            free(acc);
            free(marker);
            free(touched);
            return status;
        }

        for (int32_t i = 0; i < touched_count; i++)
        {
            int32_t col = touched[i];

            if (acc[col] != 0)
            {
                c_col_idx[nnz_count] = col;
                c_values[nnz_count] = acc[col];
                nnz_count++;
            }
        }
    }

    c_row_ptr[a_rows] = nnz_count;

    int32_t *final_col_idx = (int32_t *)realloc(
        c_col_idx,
        (size_t)nnz_count * sizeof(int32_t));

    if (final_col_idx)
    {
        c_col_idx = final_col_idx;
    }

    int32_t *final_values = (int32_t *)realloc(
        c_values,
        (size_t)nnz_count * sizeof(int32_t));

    if (final_values)
    {
        c_values = final_values;
    }

    free(acc);
    free(marker);
    free(touched);

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = nnz_count;

    return RVSP_SUCCESS;
}