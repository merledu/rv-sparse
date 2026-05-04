#include "rv_sparse/spmm_csr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dense_matmul(const float *A, const float *B, int M, int K, int N,
                         float *C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

static int dense_to_csr(const float *dense, int rows, int cols, int *rp,
                        int *ci, float *vals)
{
    int nnz = 0;
    rp[0] = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float v = dense[r * cols + c];
            if (v != 0.f) {
                ci[nnz] = c;
                vals[nnz] = v;
                nnz++;
            }
        }
        rp[r + 1] = nnz;
    }
    return nnz;
}

static int csr_dense_agrees(int M, int N, const int *cR, const int *cC,
                            const float *cD, const float *dense_ref, float eps)
{
    for (int r = 0; r < M; r++) {
        float *row = (float *)calloc((size_t)N, sizeof(float));
        if (!row) {
            return 0;
        }
        for (int p = cR[r]; p < cR[r + 1]; p++) {
            row[cC[p]] = cD[p];
        }
        for (int j = 0; j < N; j++) {
            if (fabsf(row[j] - dense_ref[r * N + j]) > eps) {
                free(row);
                return 0;
            }
        }
        free(row);
    }
    return 1;
}

static int test_known_small(void)
{
    float AD[4] = {1.f, 0.f, 0.f, 2.f};
    float BD[4] = {3.f, 0.f, 0.f, 4.f};
    int aR[3] = {0, 1, 2};
    int aC[2] = {0, 1};
    float aV[2] = {1.f, 2.f};
    int bR[3] = {0, 1, 2};
    int bC[2] = {0, 1};
    float bV[2] = {3.f, 4.f};

    float cref[4];
    dense_matmul(AD, BD, 2, 2, 2, cref);

    int cap = 8;
    float *cD = malloc((size_t)cap * sizeof(float));
    int *cC = malloc((size_t)cap * sizeof(int));
    int *cR = malloc(3 * sizeof(int));
    RowWiseInnerProduct(2, 2, 2, aV, aC, aR, bV, bC, bR, cD, cC, cR);

    int ok = csr_dense_agrees(2, 2, cR, cC, cD, cref, 1e-5f);
    free(cD);
    free(cC);
    free(cR);
    return ok;
}

static int test_rectangular(void)
{
    const int M = 3;
    const int K = 4;
    const int N = 2;
    float *Ad = malloc((size_t)M * (size_t)K * sizeof(float));
    float *Bd = malloc((size_t)K * (size_t)N * sizeof(float));
    float *Cd = malloc((size_t)M * (size_t)N * sizeof(float));
    int *aR = malloc((size_t)(M + 1) * sizeof(int));
    int *bR = malloc((size_t)(K + 1) * sizeof(int));
    int capAB = M * K + K * N;
    int *aC = malloc((size_t)capAB * sizeof(int));
    int *bC = malloc((size_t)capAB * sizeof(int));
    float *aV = malloc((size_t)capAB * sizeof(float));
    float *bV = malloc((size_t)capAB * sizeof(float));

    unsigned seed = 42u;
    for (int i = 0; i < M * K; i++) {
        seed = seed * 1664525u + 1013904223u;
        Ad[i] = (float)(seed % 7) - 3.f;
        if (seed % 11 == 0) {
            Ad[i] = 0.f;
        }
    }
    for (int i = 0; i < K * N; i++) {
        seed = seed * 1664525u + 1013904223u;
        Bd[i] = (float)(seed % 5) - 2.f;
        if (seed % 13 == 0) {
            Bd[i] = 0.f;
        }
    }

    int aNNZ = dense_to_csr(Ad, M, K, aR, aC, aV);
    int bNNZ = dense_to_csr(Bd, K, N, bR, bC, bV);
    (void)aNNZ;
    (void)bNNZ;

    dense_matmul(Ad, Bd, M, K, N, Cd);

    int capC = M * N;
    float *cD = malloc((size_t)capC * sizeof(float));
    int *cC = malloc((size_t)capC * sizeof(int));
    int *cR = malloc((size_t)(M + 1) * sizeof(int));

    RowWiseInnerProduct(M, K, N, aV, aC, aR, bV, bC, bR, cD, cC, cR);

    int ok = csr_dense_agrees(M, N, cR, cC, cD, Cd, 1e-4f);

    free(Ad);
    free(Bd);
    free(Cd);
    free(aR);
    free(bR);
    free(aC);
    free(bC);
    free(aV);
    free(bV);
    free(cD);
    free(cC);
    free(cR);
    return ok;
}

int main(void)
{
    int failed = 0;
    if (!test_known_small()) {
        fputs("FAIL: test_known_small\n", stderr);
        failed = 1;
    }
    if (!test_rectangular()) {
        fputs("FAIL: test_rectangular\n", stderr);
        failed = 1;
    }
    if (failed) {
        return 1;
    }
    puts("All tests passed.");
    return 0;
}
