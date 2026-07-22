/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Scalar FP64 CSR SpGEMM kernel.
 * Computes FP64 x FP64 with FP64 accumulation/output.
 */

#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

#include "rv_sparse.h"
#include "csr_spgemm_kernels.h"

static int compare_i32_f64(const void *a, const void *b)
{
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;

    if (ia < ib)
    {
        return -1;
    }

    if (ia > ib)
    {
        return 1;
    }

    return 0;
}

static void reset_f64_workspace(
    double *acc,
    uint8_t *mark,
    const int32_t *touched,
    int32_t touched_count)
{
    for (int32_t i = 0; i < touched_count; i++)
    {
        int32_t col = touched[i];
        acc[col] = 0.0;
        mark[col] = 0;
    }
}

rvsp_status_t rvsp_spgemm_csr_scalar_f64_raw(int32_t a_rows, int32_t a_cols, int32_t b_cols,
                                             const int32_t *a_row_ptr, const int32_t *a_col_idx,
                                             const double *a_values, const int32_t *b_row_ptr,
                                             const int32_t *b_col_idx, const double *b_values,
                                             int32_t **c_row_ptr_out, int32_t **c_col_idx_out,
                                             double **c_values_out, int32_t *c_nnz_out)
{
    if (a_rows < 0 || a_cols < 0 || b_cols < 0 ||
        a_row_ptr == NULL || a_col_idx == NULL || a_values == NULL ||
        b_row_ptr == NULL || b_col_idx == NULL || b_values == NULL ||
        c_row_ptr_out == NULL || c_col_idx_out == NULL ||
        c_values_out == NULL || c_nnz_out == NULL)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    *c_row_ptr_out = NULL;
    *c_col_idx_out = NULL;
    *c_values_out = NULL;
    *c_nnz_out = 0;

    int32_t *c_row_ptr = (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));
    double *acc = NULL;
    int32_t *touched = NULL;
    uint8_t *mark = NULL;

    if (c_row_ptr == NULL)
    {
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    if (b_cols > 0)
    {
        acc = (double *)malloc((size_t)b_cols * sizeof(double));
        touched = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));
        mark = (uint8_t *)calloc((size_t)b_cols, sizeof(uint8_t));

        if (acc == NULL || touched == NULL || mark == NULL)
        {
            free(c_row_ptr);
            free(acc);
            free(touched);
            free(mark);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }
    }

    // Symbolic pass.
    int64_t total_nnz = 0;

    for (int32_t row = 0; row < a_rows; row++)
    {
        int32_t a_start = a_row_ptr[row];
        int32_t a_end = a_row_ptr[row + 1];

        if (a_start < 0 || a_end < a_start)
        {
            free(c_row_ptr);
            free(acc);
            free(touched);
            free(mark);
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        int32_t touched_count = 0;

        for (int32_t ap = a_start; ap < a_end; ap++)
        {
            int32_t a_col = a_col_idx[ap];

            if (a_col < 0 || a_col >= a_cols)
            {
                reset_f64_workspace(acc, mark, touched, touched_count);
                free(c_row_ptr);
                free(acc);
                free(touched);
                free(mark);
                return RVSP_ERROR_INVALID_ARGUMENT;
            }

            int32_t b_start = b_row_ptr[a_col];
            int32_t b_end = b_row_ptr[a_col + 1];

            if (b_start < 0 || b_end < b_start)
            {
                reset_f64_workspace(acc, mark, touched, touched_count);
                free(c_row_ptr);
                free(acc);
                free(touched);
                free(mark);
                return RVSP_ERROR_INVALID_ARGUMENT;
            }

            for (int32_t bp = b_start; bp < b_end; bp++)
            {
                int32_t col = b_col_idx[bp];

                if (col < 0 || col >= b_cols)
                {
                    reset_f64_workspace(acc, mark, touched, touched_count);
                    free(c_row_ptr);
                    free(acc);
                    free(touched);
                    free(mark);
                    return RVSP_ERROR_INVALID_ARGUMENT;
                }

                if (mark[col] == 0)
                {
                    mark[col] = 1;
                    touched[touched_count] = col;
                    touched_count++;
                }
            }
        }

        c_row_ptr[row] = touched_count; // per-row count; prefix-summed below

        total_nnz += touched_count;

        reset_f64_workspace(acc, mark, touched, touched_count);

        if (total_nnz > INT32_MAX)
        {
            free(c_row_ptr);
            free(acc);
            free(touched);
            free(mark);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }
    }

    // Exclusive prefix sum turns the per-row counts into row pointers.
    int32_t running = 0;

    for (int32_t row = 0; row < a_rows; row++)
    {
        int32_t count = c_row_ptr[row];
        c_row_ptr[row] = running;
        running += count;
    }

    c_row_ptr[a_rows] = running;

    size_t alloc_nnz = total_nnz > 0 ? (size_t)total_nnz : 1;
    int32_t *c_col_idx = (int32_t *)malloc(alloc_nnz * sizeof(int32_t));
    double *c_values = (double *)malloc(alloc_nnz * sizeof(double));

    if (c_col_idx == NULL || c_values == NULL)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        free(acc);
        free(touched);
        free(mark);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    // Numeric pass.
    for (int32_t row = 0; row < a_rows; row++)
    {
        int32_t a_start = a_row_ptr[row];
        int32_t a_end = a_row_ptr[row + 1];
        int32_t touched_count = 0;

        for (int32_t ap = a_start; ap < a_end; ap++)
        {
            int32_t a_col = a_col_idx[ap];
            int32_t b_start = b_row_ptr[a_col];
            int32_t b_end = b_row_ptr[a_col + 1];
            double a_val = a_values[ap];

            for (int32_t bp = b_start; bp < b_end; bp++)
            {
                int32_t col = b_col_idx[bp];

                if (mark[col] == 0)
                {
                    mark[col] = 1;
                    acc[col] = 0.0;
                    touched[touched_count] = col;
                    touched_count++;
                }

                acc[col] += a_val * b_values[bp];
            }
        }

        // Keep output rows in ascending column order (canonical CSR).
        qsort(touched, (size_t)touched_count, sizeof(int32_t), compare_i32_f64);

        int32_t dst = c_row_ptr[row];

        for (int32_t i = 0; i < touched_count; i++)
        {
            int32_t col = touched[i];

            // Entries that numerically cancel to zero are not stored.
            if (acc[col] != 0.0)
            {
                c_col_idx[dst] = col;
                c_values[dst] = acc[col];
                dst++;
            }

            mark[col] = 0;
        }

        // Symbolic counts are an upper bound since cancellation can shrink rows.
        c_row_ptr[row + 1] = dst;
    }

    free(acc);
    free(touched);
    free(mark);

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = c_row_ptr[a_rows];

    return RVSP_SUCCESS;
}
