#include <stdio.h>
#include <stdlib.h>

void RowWiseInnerProduct(const int M, const int __attribute__((unused)) K, const int N,
                         float* aD, int* aC, int* aR,
                         float* bD, int* bC, int* bR,
                         float* cD, int* cC, int* cR) {
    int row_count = 0;
    for (int row = 0; row < M; row++) {
        cR[row] = row_count;

        float* cD_ = (float *)calloc(N, sizeof(float));
        if (!cD_) {
            fprintf(stderr, "Out of memory at row %d\n", row);
            exit(1);
        }

        for (int col = aR[row]; col < aR[row + 1]; col++) {
            float valA = aD[col];
            for (int b_col = bR[aC[col]]; b_col < bR[aC[col] + 1]; b_col++) {
                float valB = bD[b_col];
                int idx = bC[b_col];
                cD_[idx] += valA * valB;
            }
        }

        for (int i = 0; i < N; i++) {
            if (cD_[i] != 0.0f) {
                cC[row_count] = i;
                cD[row_count] = cD_[i];
                row_count++;
            }
        }

        free(cD_);
    }
    cR[M] = row_count;
}

static long long safe_cNNZ(int M, int N) {
    return (long long)M * N;
}

int main() {
    FILE *file, *file1, *file2, *file3;

    int gInfo[4];
    file3 = fopen("MatData/GeneralInfo.txt", "r");
    if (!file3) { perror("Error opening GeneralInfo.txt"); return -1; }
    for (int i = 0; i < 4; i++) {
        if (fscanf(file3, "%d", &gInfo[i]) != 1) {
            fprintf(stderr, "GeneralInfo.txt has fewer than 4 entries\n");
            fclose(file3);
            return -1;
        }
    }
    fclose(file3);

    const int M    = gInfo[0];
    const int K    = gInfo[1];
    const int N    = gInfo[1];
    const int aNNZ = gInfo[2];
    const int bNNZ = gInfo[2];

    float* aD = (float *)malloc(aNNZ * sizeof(float));
    int*   aC = (int   *)malloc(aNNZ * sizeof(int));
    int*   aR = (int   *)malloc((M + 1) * sizeof(int));
    float* bD = (float *)malloc(bNNZ * sizeof(float));
    int*   bC = (int   *)malloc(bNNZ * sizeof(int));
    int*   bR = (int   *)malloc((K + 1) * sizeof(int));

    if (!aD || !aC || !aR || !bD || !bC || !bR) {
        fprintf(stderr, "Out of memory allocating input matrices\n");
        return -1;
    }

    int idx = 0;
    float val;

    file = fopen("MatData/CSR_values.txt", "r");
    if (!file) { perror("Error opening CSR_values.txt"); return -1; }
    while (fscanf(file, "%f", &val) == 1) { aD[idx] = bD[idx] = val; idx++; }
    fclose(file);

    int number;
    file1 = fopen("MatData/CSR_colIdx.txt", "r");
    if (!file1) { perror("Error opening CSR_colIdx.txt"); return -1; }
    idx = 0;
    while (fscanf(file1, "%d", &number) == 1) { aC[idx] = bC[idx] = number; idx++; }
    fclose(file1);

    file2 = fopen("MatData/CSR_rowPtr.txt", "r");
    if (!file2) { perror("Error opening CSR_rowPtr.txt"); return -1; }
    idx = 0;
    while (fscanf(file2, "%d", &number) == 1) { aR[idx] = bR[idx] = number; idx++; }
    fclose(file2);

    long long cNNZ = safe_cNNZ(M, N);
    printf("Output buffer size = %lld\n", cNNZ);

    float* cD = (float *)malloc(cNNZ * sizeof(float));
    int*   cC = (int   *)malloc(cNNZ * sizeof(int));
    int*   cR = (int   *)malloc((M + 1) * sizeof(int));

    if (!cD || !cC || !cR) {
        fprintf(stderr, "Out of memory allocating output matrix\n");
        return -1;
    }

    RowWiseInnerProduct(M, K, N, aD, aC, aR, bD, bC, bR, cD, cC, cR);

    printf("Actual cNNZ = %d\n", cR[M]);

    free(aD); free(aC); free(aR);
    free(bD); free(bC); free(bR);
    free(cD); free(cC); free(cR);

    return 0;
}
