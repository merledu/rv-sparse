#include "rv_sparse/spmm_csr.h"

#include <stdlib.h>

void RowWiseInnerProduct(int M, int K, int N, const float *aD, const int *aC,
                         const int *aR, const float *bD, const int *bC,
                         const int *bR, float *cD, int *cC, int *cR)
{
    (void)K;
    int row_count = 0;
    for (int row = 0; row < M; row++) {
        cR[row] = row_count;
        float *acc = (float *)calloc((size_t)N, sizeof(float));
        if (!acc) {
            cR[M] = row_count;
            return;
        }
        for (int col = aR[row]; col < aR[row + 1]; col++) {
            float valA = aD[col];
            int brow = aC[col];
            for (int b_col = bR[brow]; b_col < bR[brow + 1]; b_col++) {
                float valB = bD[b_col];
                int idx = bC[b_col];
                acc[idx] += valA * valB;
            }
        }
        for (int i = 0; i < N; i++) {
            if (acc[i] != 0.f) {
                cC[row_count] = i;
                cD[row_count] = acc[i];
                row_count++;
            }
        }
        free(acc);
    }
    cR[M] = row_count;
}

float cNNZ_estimation(int M, int K, int N, int aNNZ, int bNNZ)
{
    long long MxK = (long long)M * (long long)K;
    float density = (float)aNNZ / (float)MxK + (float)bNNZ / (float)MxK;
    return density * (float)M * (float)N;
}
