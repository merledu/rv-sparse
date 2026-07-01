#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include "csr_spgemm_kernels.h"

static int compare_i32_f32(const void *a, const void *b)
{
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;

    if (ia < ib)
        return -1;
    if (ia > ib)
        return 1;
    return 0;
}

static int reserve_f32_output_capacity_marked(int32_t **col_idx, float **values, int32_t *capacity, int32_t required)
{
    if (required <= *capacity)
    {
        return 0;
    }

    int32_t new_capacity = (*capacity == 0) ? 16 : *capacity;

    while (new_capacity < required)
    {
        new_capacity *= 2;
    }

    int32_t *new_col_idx = (int32_t *)realloc(*col_idx, (size_t)new_capacity * sizeof(int32_t));
    if (new_col_idx == NULL)
    {
        return -1;
    }

    float *new_values = (float *)realloc(*values, (size_t)new_capacity * sizeof(float));
    if (new_values == NULL)
    {
        return -1;
    }

    *col_idx = new_col_idx;
    *values = new_values;
    *capacity = new_capacity;

    return 0;
}

rvsp_status_t rvsp_spgemm_csr_rvv_f32_indexed_marked_raw(int32_t a_rows, int32_t a_cols, int32_t b_cols,
                                                        const int32_t *a_row_ptr, const int32_t *a_col_idx,
                                                        const float *a_values, const int32_t *b_row_ptr,
                                                        const int32_t *b_col_idx, const float *b_values,
                                                        int32_t **c_row_ptr_out, int32_t **c_col_idx_out,
                                                        float **c_values_out, int32_t *c_nnz_out)
{
    if (a_rows < 0 || a_cols < 0 || b_cols < 0 ||
        a_row_ptr == NULL || a_col_idx == NULL || a_values == NULL ||
        b_row_ptr == NULL || b_col_idx == NULL || b_values == NULL ||
        c_row_ptr_out == NULL || c_col_idx_out == NULL ||
        c_values_out == NULL || c_nnz_out == NULL)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t *c_row_ptr = (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));
    float *acc = (float *)calloc((size_t)b_cols, sizeof(float));
    int32_t *marker = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));
    int32_t *touched = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));

    if (c_row_ptr == NULL || acc == NULL || marker == NULL || touched == NULL)
    {
        free(c_row_ptr);
        free(acc);
        free(marker);
        free(touched);
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    for (int32_t col = 0; col < b_cols; ++col)
    {
        marker[col] = -1;
    }

    int32_t *c_col_idx = NULL;
    float *c_values = NULL;
    int32_t capacity = 0;
    int32_t nnz_count = 0;

    for (int32_t i = 0; i < a_rows; ++i)
    {
        int32_t touched_count = 0;

        for (int32_t ap = a_row_ptr[i]; ap < a_row_ptr[i + 1]; ++ap)
        {
            int32_t a_col = a_col_idx[ap];

            if (a_col < 0 || a_col >= a_cols)
            {
                free(c_row_ptr);
                free(c_col_idx);
                free(c_values);
                free(acc);
                free(marker);
                free(touched);
                return RVSP_ERROR_INVALID_ARGUMENT;
            }

            for (int32_t bp = b_row_ptr[a_col]; bp < b_row_ptr[a_col + 1]; ++bp)
            {
                int32_t col = b_col_idx[bp];

                if (col < 0 || col >= b_cols)
                {
                    free(c_row_ptr);
                    free(c_col_idx);
                    free(c_values);
                    free(acc);
                    free(marker);
                    free(touched);
                    return RVSP_ERROR_INVALID_ARGUMENT;
                }

                if (marker[col] != i)
                {
                    marker[col] = i;
                    acc[col] = 0.0f;
                    touched[touched_count++] = col;
                }
            }
        }

        for (int32_t ap = a_row_ptr[i]; ap < a_row_ptr[i + 1]; ++ap)
        {
            int32_t a_col = a_col_idx[ap];
            float a_val = a_values[ap];

            rvsp_status_t status = rvsp_accumulate_row_f32_rvv_indexed_fast(
                a_val,
                b_row_ptr[a_col + 1] - b_row_ptr[a_col],
                &b_col_idx[b_row_ptr[a_col]],
                &b_values[b_row_ptr[a_col]],
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

        qsort(touched, (size_t)touched_count, sizeof(int32_t), compare_i32_f32);

        for (int32_t t = 0; t < touched_count; ++t)
        {
            int32_t col = touched[t];

            if (acc[col] != 0.0f)
            {
                if (reserve_f32_output_capacity_marked(
                        &c_col_idx,
                        &c_values,
                        &capacity,
                        nnz_count + 1) != 0)
                {
                    free(c_row_ptr);
                    free(c_col_idx);
                    free(c_values);
                    free(acc);
                    free(marker);
                    free(touched);
                    return RVSP_ERROR_INVALID_ARGUMENT;
                }

                c_col_idx[nnz_count] = col;
                c_values[nnz_count] = acc[col];
                nnz_count++;
            }
        }

        c_row_ptr[i + 1] = nnz_count;
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