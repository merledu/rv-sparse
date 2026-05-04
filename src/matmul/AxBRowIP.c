#include "rv_sparse/spmm_csr.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    FILE *file;
    FILE *file1;
    FILE *file2;
    FILE *file3;

    int gInfo[5];
    int number;
    file3 = fopen("matrix_data/GeneralInfo.txt", "r");
    if (file3 == NULL) {
        perror("Error opening file");
        return -1;
    }
    int idx = 0;
    while (idx < (int)(sizeof(gInfo) / sizeof(gInfo[0])) &&
           fscanf(file3, "%d", &number) == 1) {
        gInfo[idx] = number;
        idx++;
    }
    fclose(file3);
    if (idx < 4) {
        fprintf(stderr,
                "matrix_data/GeneralInfo.txt: need at least 4 integers "
                "(legacy M K aNNZ rN or extended M K N aNNZ rN)\n");
        return -1;
    }

    const int M = gInfo[0];
    const int K = gInfo[1];
    const int N = (idx >= 5) ? gInfo[2] : gInfo[0];
    const int aNNZ = (idx >= 5) ? gInfo[3] : gInfo[2];
    const int bNNZ = aNNZ;
    (void)gInfo[4];

    float *aD = (float *)malloc((size_t)aNNZ * sizeof(float));
    int *aC = (int *)malloc((size_t)aNNZ * sizeof(int));
    int *aR = (int *)malloc((size_t)(M + 1) * sizeof(int));

    float *bD = (float *)malloc((size_t)bNNZ * sizeof(float));
    int *bC = (int *)malloc((size_t)bNNZ * sizeof(int));
    int *bR = (int *)malloc((size_t)(K + 1) * sizeof(int));

    idx = 0;
    file = fopen("matrix_data/CSR_values.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }
    float val;
    while (fscanf(file, "%f", &val) == 1) {
        aD[idx] = val;
        bD[idx] = val;
        idx++;
    }
    fclose(file);

    file1 = fopen("matrix_data/CSR_colIdx.txt", "r");
    if (file1 == NULL) {
        perror("Error opening file");
        return -1;
    }
    idx = 0;
    while (fscanf(file1, "%d", &number) == 1) {
        aC[idx] = number;
        bC[idx] = number;
        idx++;
    }
    fclose(file1);

    file2 = fopen("matrix_data/CSR_rowPtr.txt", "r");
    if (file2 == NULL) {
        perror("Error opening file");
        return -1;
    }
    idx = 0;
    while (fscanf(file2, "%d", &number) == 1) {
        aR[idx] = number;
        bR[idx] = number;
        idx++;
    }
    fclose(file2);

    int est = (int)cNNZ_estimation(M, K, N, aNNZ, bNNZ);
    printf("Estimated cNNZ = %d \n", est);

    int cap = M * N;
    if (cap < est) {
        cap = est;
    }
    float *cD = (float *)malloc((size_t)cap * sizeof(float));
    int *cC = (int *)malloc((size_t)cap * sizeof(int));
    int *cR = (int *)malloc((size_t)(M + 1) * sizeof(int));

    RowWiseInnerProduct(M, K, N, aD, aC, aR, bD, bC, bR, cD, cC, cR);

    printf("Actual cNNZ = %d \n", cR[M]);

    free(aD);
    free(aC);
    free(aR);
    free(bD);
    free(bC);
    free(bR);
    free(cD);
    free(cC);
    free(cR);

    return 0;
}
