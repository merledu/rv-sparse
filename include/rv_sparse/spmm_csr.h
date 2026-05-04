#ifndef RV_SPARSE_SPMM_CSR_H
#define RV_SPARSE_SPMM_CSR_H

#ifdef __cplusplus
extern "C" {
#endif

void RowWiseInnerProduct(int M, int K, int N, const float *aD, const int *aC,
                         const int *aR, const float *bD, const int *bC,
                         const int *bR, float *cD, int *cC, int *cR);

float cNNZ_estimation(int M, int K, int N, int aNNZ, int bNNZ);

#ifdef __cplusplus
}
#endif

#endif
