/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Gustavson SpGEMM driver. Each kernel supplies its own rvsp_accum_row and
 * includes this file to generate the numeric pass and the public entry point.
 */

#ifndef RVSP_KERNEL_NAME
#error "define RVSP_KERNEL_NAME before including gustavson_core_f32.inc"
#endif

#define RVSP_CAT_(a, b) a##b
#define RVSP_CAT(a, b) RVSP_CAT_(a, b)
#define RVSP_NUMERIC_FN RVSP_CAT(RVSP_KERNEL_NAME, _numeric)

#if RVSP_TILE_COLS > 0

#define RVSP_NUMERIC_TILED_FN RVSP_CAT(RVSP_KERNEL_NAME, _numeric_tiled)

/*
 * Tiled numeric pass. Walks B in column strips of RVSP_TILE_COLS so the live
 * slice of acc[] stays cache-resident.
 *
 * Two cursor arrays make the strips cheap. b_pos[k] is how far into B row k
 * the previous strip reached; because a row's columns are ascending and the
 * strips advance monotonically, the cursors only ever move forward and their
 * total movement over the whole call is nnz(B). c_pos[i] does the same for
 * the output row, which works for the same reason: c_col_idx is sorted, so a
 * strip's share of row i is a contiguous run.
 *
 * Returns non-zero on success, 0 if the cursor allocation failed.
 */
static int RVSP_NUMERIC_TILED_FN(
    int32_t a_rows,
    int32_t b_rows,
    int32_t b_cols,
    const int32_t *RVSP_RESTRICT a_row_ptr,
    const int32_t *RVSP_RESTRICT a_col_idx,
    const float *RVSP_RESTRICT a_values,
    const int32_t *RVSP_RESTRICT b_row_ptr,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc,
    const int32_t *RVSP_RESTRICT c_row_ptr,
    const int32_t *RVSP_RESTRICT c_col_idx,
    float *RVSP_RESTRICT c_values)
{
    int32_t *b_pos = (int32_t *)malloc((size_t)b_rows * sizeof(int32_t));
    int32_t *b_end = (int32_t *)malloc((size_t)b_rows * sizeof(int32_t));
    int32_t *c_pos = (int32_t *)malloc((size_t)a_rows * sizeof(int32_t));

    if (b_pos == NULL || b_end == NULL || c_pos == NULL)
    {
        free(b_pos);
        free(b_end);
        free(c_pos);
        return 0;
    }

    for (int32_t k = 0; k < b_rows; k++)
    {
        b_pos[k] = b_row_ptr[k];
    }

    for (int32_t row = 0; row < a_rows; row++)
    {
        c_pos[row] = c_row_ptr[row];
    }

    for (int32_t t0 = 0; t0 < b_cols; t0 += RVSP_TILE_COLS)
    {
        const int32_t t1 =
            (t0 + RVSP_TILE_COLS < b_cols) ? t0 + RVSP_TILE_COLS : b_cols;

        /* Advance each B row's cursor to the end of this strip. Summed over
         * every strip this walks each B row exactly once. */
        for (int32_t k = 0; k < b_rows; k++)
        {
            const int32_t stop = b_row_ptr[k + 1];
            int32_t e = b_pos[k];

            while (e < stop && b_col_idx[e] < t1)
            {
                e++;
            }

            b_end[k] = e;
        }

        for (int32_t row = 0; row < a_rows; row++)
        {
            const int32_t a_start = a_row_ptr[row];
            const int32_t a_end = a_row_ptr[row + 1];

            for (int32_t ap = a_start; ap < a_end; ap++)
            {
                const int32_t k = a_col_idx[ap];
                const int32_t start = b_pos[k];

                rvsp_accum_row(
                    a_values[ap],
                    b_end[k] - start,
                    &b_col_idx[start],
                    &b_values[start],
                    acc);
            }

            /* This strip's share of row i is the contiguous run of C entries
             * whose column is below t1, picking up where the last strip
             * stopped. */
            const int32_t c_stop = c_row_ptr[row + 1];
            int32_t j = c_pos[row];

            while (j < c_stop && c_col_idx[j] < t1)
            {
                const int32_t col = c_col_idx[j];

                c_values[j] = acc[col];
                acc[col] = 0.0f;
                j++;
            }

            c_pos[row] = j;
        }

        for (int32_t k = 0; k < b_rows; k++)
        {
            b_pos[k] = b_end[k];
        }
    }

    free(b_pos);
    free(b_end);
    free(c_pos);

    return 1;
}

#endif /* RVSP_TILE_COLS > 0 */

void RVSP_NUMERIC_FN(
    int32_t a_rows,
    const int32_t *RVSP_RESTRICT a_row_ptr,
    const int32_t *RVSP_RESTRICT a_col_idx,
    const float *RVSP_RESTRICT a_values,
    const int32_t *RVSP_RESTRICT b_row_ptr,
    const int32_t *RVSP_RESTRICT b_col_idx,
    const float *RVSP_RESTRICT b_values,
    float *RVSP_RESTRICT acc,
    const int32_t *RVSP_RESTRICT c_row_ptr,
    const int32_t *RVSP_RESTRICT c_col_idx,
    float *RVSP_RESTRICT c_values)
{
    for (int32_t row = 0; row < a_rows; row++)
    {
        const int32_t a_start = a_row_ptr[row];
        const int32_t a_end = a_row_ptr[row + 1];

        for (int32_t ap = a_start; ap < a_end; ap++)
        {
            const int32_t k = a_col_idx[ap];
            const float a_val = a_values[ap];
            const int32_t b_start = b_row_ptr[k];
            const int32_t b_end = b_row_ptr[k + 1];

            rvsp_accum_row(
                a_val,
                b_end - b_start,
                &b_col_idx[b_start],
                &b_values[b_start],
                acc);
        }

        /* Drain the touched acc[] slots into C and clear them for the next
         * row, so acc stays zeroed without a full sweep. */
        const int32_t c_start = c_row_ptr[row];
        const int32_t c_end = c_row_ptr[row + 1];

        for (int32_t i = c_start; i < c_end; i++)
        {
            const int32_t col = c_col_idx[i];

            c_values[i] = acc[col];
            acc[col] = 0.0f;
        }
    }
}

rvsp_status_t RVSP_KERNEL_NAME(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *a_row_ptr,
    const int32_t *a_col_idx,
    const float *a_values,
    const int32_t *b_row_ptr,
    const int32_t *b_col_idx,
    const float *b_values,
    void *workspace,
    size_t workspace_bytes,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    float **c_values_out,
    int32_t *c_nnz_out,
    int64_t *op_counts_out)
{
#if RVSP_TILE_COLS == 0
    (void)a_cols;
#endif

    if (a_row_ptr == NULL || a_col_idx == NULL || a_values == NULL ||
        b_row_ptr == NULL || b_col_idx == NULL || b_values == NULL ||
        c_row_ptr_out == NULL || c_col_idx_out == NULL ||
        c_values_out == NULL || c_nnz_out == NULL || workspace == NULL)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (a_rows <= 0 || a_cols <= 0 || b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    if (workspace_bytes < rvsp_ws_bytes(b_cols))
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    *c_row_ptr_out = NULL;
    *c_col_idx_out = NULL;
    *c_values_out = NULL;
    *c_nnz_out = 0;

    rvsp_ws_t ws;
    rvsp_ws_bind(&ws, workspace, b_cols);

    memset(ws.acc, 0, (size_t)b_cols * sizeof(float));
    memset(ws.mark, 0, (size_t)b_cols * sizeof(uint8_t));

    int32_t *c_row_ptr =
        (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));

    if (c_row_ptr == NULL)
    {
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    int64_t total_nnz = 0;

    /* Symbolic pass first, so C can be sized exactly before the numeric pass. */
    rvsp_status_t status =
        rvsp_symbolic_count(
            a_rows,
            a_row_ptr,
            a_col_idx,
            b_row_ptr,
            b_col_idx,
            ws.mark,
            ws.touched,
            c_row_ptr,
            op_counts_out,
            &total_nnz);

    if (status != RVSP_SUCCESS)
    {
        free(c_row_ptr);
        return status;
    }

    /* malloc(0) is implementation-defined, so floor the allocation at 1. */
    const size_t alloc_nnz =
        total_nnz > 0 ? (size_t)total_nnz : 1;

    int32_t *c_col_idx =
        (int32_t *)malloc(alloc_nnz * sizeof(int32_t));

    float *c_values =
        (float *)malloc(alloc_nnz * sizeof(float));

    if (c_col_idx == NULL || c_values == NULL)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    rvsp_symbolic_fill(
        a_rows,
        b_cols,
        a_row_ptr,
        a_col_idx,
        b_row_ptr,
        b_col_idx,
        ws.mark,
        ws.scratch,
        c_row_ptr,
        c_col_idx);

    int numeric_done = 0;

#if RVSP_TILE_COLS > 0
    /* b_rows is a_cols: A's columns index B's rows. */
    numeric_done =
        RVSP_NUMERIC_TILED_FN(
            a_rows,
            a_cols,
            b_cols,
            a_row_ptr,
            a_col_idx,
            a_values,
            b_row_ptr,
            b_col_idx,
            b_values,
            ws.acc,
            c_row_ptr,
            c_col_idx,
            c_values);
#endif

    /* Falls through when tiling is off, and when its cursor allocation
     * failed. acc[] is untouched in the failure case, so the full-width pass
     * is still safe to run. */
    if (!numeric_done)
    {
        RVSP_NUMERIC_FN(
            a_rows,
            a_row_ptr,
            a_col_idx,
            a_values,
            b_row_ptr,
            b_col_idx,
            b_values,
            ws.acc,
            c_row_ptr,
            c_col_idx,
            c_values);
    }

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = c_row_ptr[a_rows];

    return RVSP_SUCCESS;
}