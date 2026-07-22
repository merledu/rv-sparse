#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

#include "csr_spgemm_kernels.h"

static int compare_i32_rvv_f64(const void *a, const void *b)
{
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;

    if (ia < ib) {
        return -1;
    }

    if (ia > ib) {
        return 1;
    }

    return 0;
}

static void clear_f64_workspace(
    double *acc,
    uint8_t *mark,
    const int32_t *touched,
    int32_t touched_count
)
{
    for (int32_t i = 0; i < touched_count; i++) {
        int32_t col = touched[i];
        acc[col] = 0.0;
        mark[col] = 0;
    }
}

static rvsp_status_t mark_f64_row_columns(
    int32_t b_nnz,
    const int32_t *b_col_idx,
    int32_t b_cols,
    double *acc,
    uint8_t *mark,
    int32_t *touched,
    int32_t *touched_count
)
{
    for (int32_t p = 0; p < b_nnz; p++) {
        int32_t col = b_col_idx[p];

        if (col < 0 || col >= b_cols) {
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        if (mark[col] == 0) {
            mark[col] = 1;
            touched[*touched_count] = col;
            (*touched_count)++;
            acc[col] = 0.0;
        }
    }

    return RVSP_SUCCESS;
}

static rvsp_status_t accumulate_f64_row_scalar_marked(
    double a_val,
    int32_t b_nnz,
    const int32_t *b_col_idx,
    const double *b_values,
    int32_t b_cols,
    double *acc,
    uint8_t *mark,
    int32_t *touched,
    int32_t *touched_count
)
{
    for (int32_t p = 0; p < b_nnz; p++) {
        int32_t col = b_col_idx[p];

        if (col < 0 || col >= b_cols) {
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        if (mark[col] == 0) {
            mark[col] = 1;
            touched[*touched_count] = col;
            (*touched_count)++;
            acc[col] = 0.0;
        }

        acc[col] += a_val * b_values[p];
    }

    return RVSP_SUCCESS;
}

static rvsp_status_t build_b_duplicate_flags_f64(
    int32_t b_rows,
    int32_t b_cols,
    const int32_t *b_row_ptr,
    const int32_t *b_col_idx,
    uint8_t *b_has_duplicates
)
{
    uint8_t *seen = NULL;
    int32_t *seen_touched = NULL;

    if (b_cols > 0) {
        seen = (uint8_t *)calloc((size_t)b_cols, sizeof(uint8_t));
        seen_touched = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));

        if (seen == NULL || seen_touched == NULL) {
            free(seen);
            free(seen_touched);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }
    }

    for (int32_t row = 0; row < b_rows; row++) {
        int32_t start = b_row_ptr[row];
        int32_t end = b_row_ptr[row + 1];

        if (start < 0 || end < start) {
            free(seen);
            free(seen_touched);
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        int32_t seen_count = 0;

        for (int32_t p = start; p < end; p++) {
            int32_t col = b_col_idx[p];

            if (col < 0 || col >= b_cols) {
                for (int32_t i = 0; i < seen_count; i++) {
                    seen[seen_touched[i]] = 0;
                }

                free(seen);
                free(seen_touched);
                return RVSP_ERROR_INVALID_ARGUMENT;
            }

            if (seen[col] != 0) {
                b_has_duplicates[row] = 1;
            } else {
                seen[col] = 1;
                seen_touched[seen_count] = col;
                seen_count++;
            }
        }

        for (int32_t i = 0; i < seen_count; i++) {
            seen[seen_touched[i]] = 0;
        }
    }

    free(seen);
    free(seen_touched);

    return RVSP_SUCCESS;
}

static rvsp_status_t accumulate_f64_row_rvv_or_scalar(
    double a_val,
    int32_t b_nnz,
    const int32_t *b_col_idx,
    const double *b_values,
    int32_t b_cols,
    double *acc,
    uint8_t *mark,
    int32_t *touched,
    int32_t *touched_count,
    uint8_t has_duplicates
)
{
    if (has_duplicates) {
        return accumulate_f64_row_scalar_marked(
            a_val,
            b_nnz,
            b_col_idx,
            b_values,
            b_cols,
            acc,
            mark,
            touched,
            touched_count
        );
    }

    rvsp_status_t status = mark_f64_row_columns(
        b_nnz,
        b_col_idx,
        b_cols,
        acc,
        mark,
        touched,
        touched_count
    );

    if (status != RVSP_SUCCESS) {
        return status;
    }

    return rvsp_accumulate_row_f64_rvv_indexed_fast(
        a_val,
        b_nnz,
        b_col_idx,
        b_values,
        acc
    );
}

rvsp_status_t rvsp_spgemm_csr_rvv_f64_indexed_marked_raw(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *a_row_ptr,
    const int32_t *a_col_idx,
    const double *a_values,
    const int32_t *b_row_ptr,
    const int32_t *b_col_idx,
    const double *b_values,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    double **c_values_out,
    int32_t *c_nnz_out
)
{
    if (a_rows < 0 || a_cols < 0 || b_cols < 0 ||
        a_row_ptr == NULL || a_col_idx == NULL || a_values == NULL ||
        b_row_ptr == NULL || b_col_idx == NULL || b_values == NULL ||
        c_row_ptr_out == NULL || c_col_idx_out == NULL ||
        c_values_out == NULL || c_nnz_out == NULL) {
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
    uint8_t *b_has_duplicates = NULL;

    if (c_row_ptr == NULL) {
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    if (b_cols > 0) {
        acc = (double *)malloc((size_t)b_cols * sizeof(double));
        touched = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));
        mark = (uint8_t *)calloc((size_t)b_cols, sizeof(uint8_t));

        if (acc == NULL || touched == NULL || mark == NULL) {
            free(c_row_ptr);
            free(acc);
            free(touched);
            free(mark);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }
    }

    if (a_cols > 0) {
        b_has_duplicates = (uint8_t *)calloc((size_t)a_cols, sizeof(uint8_t));

        if (b_has_duplicates == NULL) {
            free(c_row_ptr);
            free(acc);
            free(touched);
            free(mark);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }

        rvsp_status_t status = build_b_duplicate_flags_f64(
            a_cols,
            b_cols,
            b_row_ptr,
            b_col_idx,
            b_has_duplicates
        );

        if (status != RVSP_SUCCESS) {
            free(c_row_ptr);
            free(acc);
            free(touched);
            free(mark);
            free(b_has_duplicates);
            return status;
        }
    }

    // Symbolic pass: count the distinct output columns of each row,
    // validating A's and B's column indices along the way.
    int64_t total_nnz = 0;

    for (int32_t row = 0; row < a_rows; row++) {
        int32_t a_start = a_row_ptr[row];
        int32_t a_end = a_row_ptr[row + 1];

        if (a_start < 0 || a_end < a_start) {
            free(c_row_ptr);
            free(acc);
            free(touched);
            free(mark);
            free(b_has_duplicates);
            return RVSP_ERROR_INVALID_ARGUMENT;
        }

        int32_t touched_count = 0;

        for (int32_t ap = a_start; ap < a_end; ap++) {
            int32_t a_col = a_col_idx[ap];

            if (a_col < 0 || a_col >= a_cols) {
                clear_f64_workspace(acc, mark, touched, touched_count);
                free(c_row_ptr);
                free(acc);
                free(touched);
                free(mark);
                free(b_has_duplicates);
                return RVSP_ERROR_INVALID_ARGUMENT;
            }

            int32_t b_start = b_row_ptr[a_col];
            int32_t b_end = b_row_ptr[a_col + 1];

            if (b_start < 0 || b_end < b_start) {
                clear_f64_workspace(acc, mark, touched, touched_count);
                free(c_row_ptr);
                free(acc);
                free(touched);
                free(mark);
                free(b_has_duplicates);
                return RVSP_ERROR_INVALID_ARGUMENT;
            }

            rvsp_status_t status = mark_f64_row_columns(
                b_end - b_start,
                &b_col_idx[b_start],
                b_cols,
                acc,
                mark,
                touched,
                &touched_count
            );

            if (status != RVSP_SUCCESS) {
                clear_f64_workspace(acc, mark, touched, touched_count);
                free(c_row_ptr);
                free(acc);
                free(touched);
                free(mark);
                free(b_has_duplicates);
                return status;
            }
        }

        c_row_ptr[row] = touched_count; // per-row count; prefix-summed below

        total_nnz += touched_count;

        clear_f64_workspace(acc, mark, touched, touched_count);

        if (total_nnz > INT32_MAX) {
            free(c_row_ptr);
            free(acc);
            free(touched);
            free(mark);
            free(b_has_duplicates);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }
    }

    // Exclusive prefix sum turns the per-row counts into row pointers.
    int32_t running = 0;

    for (int32_t row = 0; row < a_rows; row++) {
        int32_t count = c_row_ptr[row];
        c_row_ptr[row] = running;
        running += count;
    }

    c_row_ptr[a_rows] = running;

    size_t alloc_nnz = total_nnz > 0 ? (size_t)total_nnz : 1;
    int32_t *c_col_idx = (int32_t *)malloc(alloc_nnz * sizeof(int32_t));
    double *c_values = (double *)malloc(alloc_nnz * sizeof(double));

    if (c_col_idx == NULL || c_values == NULL) {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        free(acc);
        free(touched);
        free(mark);
        free(b_has_duplicates);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    // Numeric pass.
    for (int32_t row = 0; row < a_rows; row++) {
        int32_t a_start = a_row_ptr[row];
        int32_t a_end = a_row_ptr[row + 1];
        int32_t touched_count = 0;

        for (int32_t ap = a_start; ap < a_end; ap++) {
            int32_t a_col = a_col_idx[ap];
            int32_t b_start = b_row_ptr[a_col];
            int32_t b_end = b_row_ptr[a_col + 1];

            rvsp_status_t status = accumulate_f64_row_rvv_or_scalar(
                a_values[ap],
                b_end - b_start,
                &b_col_idx[b_start],
                &b_values[b_start],
                b_cols,
                acc,
                mark,
                touched,
                &touched_count,
                b_has_duplicates ? b_has_duplicates[a_col] : 0
            );

            if (status != RVSP_SUCCESS) {
                clear_f64_workspace(acc, mark, touched, touched_count);
                free(c_row_ptr);
                free(c_col_idx);
                free(c_values);
                free(acc);
                free(touched);
                free(mark);
                free(b_has_duplicates);
                return status;
            }
        }

        // Keep output rows in ascending column order (canonical CSR).
        qsort(touched, (size_t)touched_count, sizeof(int32_t), compare_i32_rvv_f64);

        int32_t dst = c_row_ptr[row];

        for (int32_t i = 0; i < touched_count; i++) {
            int32_t col = touched[i];

            // Entries that numerically cancel to zero are not stored.
            if (acc[col] != 0.0) {
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
    free(b_has_duplicates);

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = c_row_ptr[a_rows];

    return RVSP_SUCCESS;
}
