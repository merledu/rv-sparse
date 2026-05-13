#include <stdio.h>
#include <stdlib.h>

void RowWiseInnerProduct(const int M, const int K, const int N, float* aD, int* aC, int* aR, float* bD, int* bC, int* bR, float* cD, int* cC, int* cR){
    (void)N;
    int row_count = 0;
    float* cD_ = (float *)calloc(K, sizeof(float));
    int* touched = (int *)calloc(K, sizeof(int));
    int* touched_cols = (int *)malloc(K * sizeof(int));

    for(int row=0; row<M; row++){
        cR[row] = row_count;
        int touched_count = 0;

        for(int col=aR[row]; col<aR[row+1]; col++){
            float valA = aD[col];
            for(int b_col=bR[aC[col]]; b_col<bR[aC[col]+1]; b_col++){
                float valB = bD[b_col];
                // int idx = row*K + bC[b_col];
                int idx = bC[b_col];

                if (!touched[idx]){
                    touched[idx] = 1;
                    touched_cols[touched_count] = idx;
                    touched_count++;
                }

                cD_[idx] += valA * valB;
            }
        }

        for (int i=0; i<touched_count; i++){
            int idx = touched_cols[i];

            if (cD_[idx] != 0){
                cC[row_count] = idx;
                cD[row_count] = cD_[idx];
                row_count++;
            }

            cD_[idx] = 0;
            touched[idx] = 0;
        }
    }

    free(cD_);
    free(touched);
    free(touched_cols);
    cR[M] = row_count;
}

float cNNZ_estimation(int M, int K, int N, int aNNZ, int bNNZ){
    long long MxK = (long long)M*(long long)K;
    float density = (float)(aNNZ)/MxK + (float)(bNNZ)/MxK;
    return density * M * N;
}

int main(){
    FILE *file;
    FILE *file1;
    FILE *file2;
    FILE *file3;

    int* gInfo = (int *)malloc(4*sizeof(int));
    //////////////////////////////////////////
    int number;
    file3 = fopen("matrix_data/GeneralInfo.txt", "r");
    if (file3 == NULL) {
        perror("Error opening file");
        return -1;
    }
    int idx = 0;
    while (fscanf(file3, "%d", &number) == 1) {
        gInfo[idx] = number;
        idx++;
    }
    fclose(file3);
    //////////////////////////////////////////

    const int M    = gInfo[0];
    const int K    = gInfo[1];
    const int N    = gInfo[0];
    const int aNNZ = gInfo[2];
    const int bNNZ = gInfo[2];
    const int rN   = gInfo[3];
    (void)rN;

    float* aD = (float *)malloc(aNNZ*sizeof(float));
    int*   aC = (int *)malloc(aNNZ*sizeof(int));
    int*   aR = (int *)malloc((M+1)*sizeof(int));

    float* bD = (float *)malloc(bNNZ*sizeof(float));
    int*   bC = (int *)malloc(bNNZ*sizeof(int));
    int*   bR = (int *)malloc((K+1)*sizeof(int));

    //////////////////////////////////////////
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

    //////////////////////////////////////////
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
    //////////////////////////////////////////
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
    //////////////////////////////////////////

    int cNNZ = cNNZ_estimation(M,K,N,aNNZ,bNNZ);
    printf("Estimated cNNZ = %d \n", cNNZ);

    float* cD = (float *)malloc(cNNZ*sizeof(float));
    int* cC = (int *)malloc(cNNZ*sizeof(int));
    int* cR = (int *)malloc((M+1)*sizeof(int));

    RowWiseInnerProduct(M, K, N, aD, aC, aR, bD, bC, bR, cD, cC, cR);

    // for(int i=0; i<M+1; i++){
    //     printf("%d\n", cR[i]);
    //     for(int j=cR[i]; j<cR[i+1]; j++){
    //         printf("%d %.f\n", cC[j], cD[j]);
    //     }
    // }

    
    // for(int i=0; i<M+1; i++){
    //     for(int j=cR[i]; j<cR[i+1]; j++){
    //         printf("%.f ", cD[j]);
    //     }
    //     printf("\n");
    // }

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
