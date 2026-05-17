#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../include/rv_sparse.h"

static int test_known_matrix()
{
    // Hand-built 4x4 matrix:
    // | 2 0 0 5 |
    // | 0 3 0 0 |
    // | 1 0 4 0 |
    // | 0 0 0 7 |
    CSRMatrix_f32 *A = csr_alloc_f32(4, 4, 6);
    float vals[] = {2, 5, 3, 1, 4, 7};
    int cols[] = {0, 3, 1, 0, 2, 3};
    int rowptr[] = {0, 2, 3, 5, 6};
    for (int i = 0; i < 6; i++)
    {
        A->values[i] = vals[i];
        A->col_idx[i] = cols[i];
    }
    for (int i = 0; i < 5; i++)
        A->row_ptr[i] = rowptr[i];

    float x[] = {1, 2, 3, 4};
    float y[4] = {0};
    // Expected: y = [22, 6, 13, 28]
    spmv_csr_scalar_f32(A, x, y);

    float expected[] = {22.0f, 6.0f, 13.0f, 28.0f};
    for (int i = 0; i < 4; i++)
    {
        if (fabsf(y[i] - expected[i]) > 1e-4f)
        {
            printf("FAIL: known_matrix y[%d]=%.4f expected %.4f\n",
                   i, y[i], expected[i]);
            csr_free_f32(A);
            return 0;
        }
    }
    csr_free_f32(A);
    return 1;
}

static int test_random_scalar(int rows, int cols, float density, int seed)
{
    CSRMatrix_f32 *A = generate_random_csr_f32(rows, cols, density, seed);
    float *x = malloc(cols * sizeof(float));
    float *y = malloc(rows * sizeof(float));
    for (int i = 0; i < cols; i++)
        x[i] = (float)i * 0.1f;

    spmv_csr_scalar_f32(A, x, y);

    for (int r = 0; r < rows; r++)
    {
        float ref = 0;
        for (int i = A->row_ptr[r]; i < A->row_ptr[r + 1]; i++)
            ref += A->values[i] * x[A->col_idx[i]];
        if (fabsf(y[r] - ref) > 1e-3f)
        {
            printf("FAIL: random scalar y[%d]=%.6f ref=%.6f\n", r, y[r], ref);
            free(x);
            free(y);
            csr_free_f32(A);
            return 0;
        }
    }
    // free(x);
    // free(y);
    // csr_free_f32(A);
    return 1;
}

static int test_rvv_matches_scalar(int rows, int cols,
                                   float density, int seed)
{
    CSRMatrix_f32 *A = generate_random_csr_f32(rows, cols, density, seed);
    float *x = malloc(cols * sizeof(float));
    float *y_scalar = malloc(rows * sizeof(float));
    float *y_rvv = malloc(rows * sizeof(float));

    for (int i = 0; i < cols; i++)
        x[i] = (float)i * 0.1f;

    spmv_csr_scalar_f32(A, x, y_scalar);
    spmv_csr_rvv_f32(A, x, y_rvv);

    int ok = 1;
    for (int r = 0; r < rows; r++)
    {
        if (fabsf(y_scalar[r] - y_rvv[r]) > 1e-3f)
        {
            printf("  MISMATCH row %d: scalar=%.6f rvv=%.6f\n",
                   r, y_scalar[r], y_rvv[r]);
            ok = 0;
            break;
        }
    }
    // free(x);
    // free(y_scalar);
    // free(y_rvv);
    // csr_free_f32(A);
    return ok;
}

static void csr_to_dense_f32(const CSRMatrix_f32 *A, float *dense)
{
    int rows = A->num_rows;
    int cols = A->num_cols;
    for (int r = 0; r < rows; r++)
    {
        for (int i = A->row_ptr[r]; i < A->row_ptr[r + 1]; i++)
        {
            int c = A->col_idx[i];
            dense[(size_t)r * cols + c] = A->values[i];
        }
    }
}

static int dense_equal_f32(const float *a, const float *b, int rows, int cols, float eps)
{
    int total = rows * cols;
    for (int i = 0; i < total; i++)
    {
        if (fabsf(a[i] - b[i]) > eps)
            return 0;
    }
    return 1;
}

static int test_spmm_scalar()
{
    CSRMatrix_f32 *A = csr_alloc_f32(2, 3, 3);
    A->values[0] = 1.0f;
    A->col_idx[0] = 0;
    A->values[1] = 2.0f;
    A->col_idx[1] = 2;
    A->values[2] = 3.0f;
    A->col_idx[2] = 1;
    A->row_ptr[0] = 0;
    A->row_ptr[1] = 2;
    A->row_ptr[2] = 3;

    float B[] = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
    };
    float C[4] = {0};
    float expected[] = {11.0f, 14.0f, 9.0f, 12.0f};

    spmm_csr_scalar_f32(A, B, 2, C);

    int ok = dense_equal_f32(C, expected, 2, 2, 1e-4f);
    // csr_free_f32(A);
    return ok;
}

int main()
{
    int pass = 1;
    pass &= test_known_matrix();
    printf("known_matrix:     %s\n", pass ? "PASS" : "FAIL");

    int r = test_random_scalar(100, 100, 0.05f, 42);
    printf("random_100x100:   %s\n", r ? "PASS" : "FAIL");
    pass &= r;

    r = test_random_scalar(1000, 1000, 0.01f, 99);
    printf("random_1000x1000: %s\n", r ? "PASS" : "FAIL");
    pass &= r;

    r = test_rvv_matches_scalar(4, 4, 1.00f, 1);
    printf("rvv_4x4_dense:    %s\n", r ? "PASS" : "FAIL");
    pass &= r;

    r = test_rvv_matches_scalar(100, 100, 0.05f, 42);
    printf("rvv_100x100:      %s\n", r ? "PASS" : "FAIL");
    pass &= r;

    r = test_rvv_matches_scalar(1000, 1000, 0.01f, 99);
    printf("rvv_1000x1000:    %s\n", r ? "PASS" : "FAIL");
    pass &= r;

    r = test_rvv_matches_scalar(500, 500, 0.10f, 7);
    printf("rvv_500x500:      %s\n", r ? "PASS" : "FAIL");
    pass &= r;

    r = test_spmm_scalar();
    printf("spmm_scalar:      %s\n", r ? "PASS" : "FAIL");
    pass &= r;

    return pass ? 0 : 1;
}
