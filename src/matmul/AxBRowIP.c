#include <stdio.h>
#include <stdlib.h>

void RowWiseInnerProduct(const int M, const int K, const int N, float* aD, int* aC, int* aR, float* bD, int* bC, int* bR, float* cD, int* cC, int* cR){
    int row_count = 0;
    for(int row=0; row<M; row++){
        cR[row] = row_count;
        // The accumulator size must match the column space of B (N), not K!
        float* cD_ = (float *)calloc(N, sizeof(float));
        for(int col=aR[row]; col<aR[row+1]; col++){
            float valA = aD[col];
            for(int b_col=bR[aC[col]]; b_col<bR[aC[col]+1]; b_col++){
                float valB = bD[b_col];
                int idx = bC[b_col];
                cD_[idx] += valA * valB;
            }
        }
        // Iterate through N (B's column space) instead of K!
        for (int i=0; i<N; i++){
            if (cD_[i] != 0.0f){
                cC[row_count] = i;
                cD[row_count] = cD_[i];
                row_count++;
            }
        }
        free(cD_);
    }
    cR[M] = row_count;
}

float cNNZ_estimation(int M, int K, int N, int aNNZ, int bNNZ){
    long long MxK = (long long)M*(long long)K;
    long long KxN = (long long)K*(long long)N;
    float densityA = MxK > 0 ? (float)aNNZ / MxK : 0.0f;
    float densityB = KxN > 0 ? (float)bNNZ / KxN : 0.0f;
    float density = densityA + densityB;
    if (density > 1.0f) density = 1.0f;
    float est = density * M * N;

    // Add safe lower and upper bounds
    long long max_possible = (long long)M * (long long)N;
    if (est < aNNZ) est = (float)aNNZ;
    if (est < bNNZ) est = (float)bNNZ;
    if (est > max_possible) est = (float)max_possible;
    if (est < 10.0f) est = 10.0f;
    return est;
}

int main(){
    FILE *file;
    FILE *file1;
    FILE *file2;
    FILE *file3;

    int* gInfo = (int *)malloc(5 * sizeof(int));
    //////////////////////////////////////////
    int number;
    file3 = fopen("matrix_data/GeneralInfo.txt", "r");
    if (file3 == NULL) {
        perror("Error opening matrix_data/GeneralInfo.txt");
        free(gInfo);
        return -1;
    }
    int idx = 0;
    while (fscanf(file3, "%d", &number) == 1 && idx < 5) {
        gInfo[idx] = number;
        idx++;
    }
    fclose(file3);
    //////////////////////////////////////////

    const int M    = gInfo[0];
    const int K    = gInfo[1];
    const int N    = gInfo[2];
    const int aNNZ = gInfo[3];
    const int bNNZ = gInfo[4];

    printf("Matrix A: %d x %d, NNZ = %d\n", M, K, aNNZ);
    printf("Matrix B: %d x %d, NNZ = %d\n", K, N, bNNZ);

    float* aD = (float *)malloc(aNNZ*sizeof(float));
    int*   aC = (int *)malloc(aNNZ*sizeof(int));
    int*   aR = (int *)malloc((M+1)*sizeof(int));

    float* bD = (float *)malloc(bNNZ*sizeof(float));
    int*   bC = (int *)malloc(bNNZ*sizeof(int));
    int*   bR = (int *)malloc((K+1)*sizeof(int));

    //////////////////////////////////////////
    // Load Matrix A CSR
    idx = 0;
    file = fopen("matrix_data/A_values.txt", "r");
    if (file == NULL) {
        perror("Error opening matrix_data/A_values.txt");
        return -1;
    }
    float val;
    while (fscanf(file, "%f", &val) == 1 && idx < aNNZ) {
        aD[idx] = val;
        idx++;
    }
    fclose(file);

    file1 = fopen("matrix_data/A_colIdx.txt", "r");
    if (file1 == NULL) {
        perror("Error opening matrix_data/A_colIdx.txt");
        return -1;
    }
    idx = 0;
    while (fscanf(file1, "%d", &number) == 1 && idx < aNNZ) {
        aC[idx] = number;
        idx++;
    }
    fclose(file1);

    file2 = fopen("matrix_data/A_rowPtr.txt", "r");
    if (file2 == NULL) {
        perror("Error opening matrix_data/A_rowPtr.txt");
        return -1;
    }
    idx = 0;
    while (fscanf(file2, "%d", &number) == 1 && idx < M + 1) {
        aR[idx] = number;
        idx++;
    }
    fclose(file2);

    //////////////////////////////////////////
    // Load Matrix B CSR
    idx = 0;
    file = fopen("matrix_data/B_values.txt", "r");
    if (file == NULL) {
        perror("Error opening matrix_data/B_values.txt");
        return -1;
    }
    while (fscanf(file, "%f", &val) == 1 && idx < bNNZ) {
        bD[idx] = val;
        idx++;
    }
    fclose(file);

    file1 = fopen("matrix_data/B_colIdx.txt", "r");
    if (file1 == NULL) {
        perror("Error opening matrix_data/B_colIdx.txt");
        return -1;
    }
    idx = 0;
    while (fscanf(file1, "%d", &number) == 1 && idx < bNNZ) {
        bC[idx] = number;
        idx++;
    }
    fclose(file1);

    file2 = fopen("matrix_data/B_rowPtr.txt", "r");
    if (file2 == NULL) {
        perror("Error opening matrix_data/B_rowPtr.txt");
        return -1;
    }
    idx = 0;
    while (fscanf(file2, "%d", &number) == 1 && idx < K + 1) {
        bR[idx] = number;
        idx++;
    }
    fclose(file2);
    //////////////////////////////////////////

    int cNNZ = (int)cNNZ_estimation(M,K,N,aNNZ,bNNZ);
    printf("Estimated cNNZ = %d \n", cNNZ);

    float* cD = (float *)malloc(cNNZ*sizeof(float));
    int* cC = (int *)malloc(cNNZ*sizeof(int));
    int* cR = (int *)malloc((M+1)*sizeof(int));

    RowWiseInnerProduct(M, K, N, aD, aC, aR, bD, bC, bR, cD, cC, cR);

    printf("Actual cNNZ = %d \n", cR[M]);

    printf("Resulting Matrix C (non-zero entries):\n");
    for(int i=0; i<M; i++){
        for(int j=cR[i]; j<cR[i+1]; j++){
            printf("C[%d][%d] = %.2f\n", i, cC[j], cD[j]);
        }
    }

    free(gInfo);
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