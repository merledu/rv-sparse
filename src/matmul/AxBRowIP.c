#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

/* Returns 0 on success, -1 on output-buffer overflow (cNNZ underestimate).
 * cD/cC capacity is cD_capacity entries; cR must hold M+1 entries. */
int RowWiseInnerProduct(const int M, const int K, const int N,
                        float* aD, int* aC, int* aR,
                        float* bD, int* bC, int* bR,
                        float* cD, int* cC, int* cR,
                        size_t cD_capacity){
    (void)K;
    int row_count = 0;
    /* Accumulator must span the output column range (0..N-1), not K.
     * Previously sized K, which silently overflowed whenever N > K. */
    float* cD_ = (float *)calloc((size_t)N, sizeof(float));
    if (cD_ == NULL) return -1;

    for(int row=0; row<M; row++){
        cR[row] = row_count;
        for(int col=aR[row]; col<aR[row+1]; col++){
            float valA = aD[col];
            for(int b_col=bR[aC[col]]; b_col<bR[aC[col]+1]; b_col++){
                float valB = bD[b_col];
                int idx = bC[b_col];
                cD_[idx] += valA * valB;
            }
        }
        for (int i=0; i<N; i++){
            if (cD_[i] != 0){
                if ((size_t)row_count >= cD_capacity) {
                    free(cD_);
                    cR[M] = row_count;
                    return -1;
                }
                cC[row_count] = i;
                cD[row_count] = cD_[i];
                row_count++;
                cD_[i] = 0.0f;  /* selective zero for next row */
            }
        }
    }
    free(cD_);
    cR[M] = row_count;
    return 0;
}

/* Standard sparse-product NNZ estimate: E[nnz(C)] = aNNZ * bNNZ / K
 * (uniform-density approximation of M*N*(1-(1-rho_a*rho_b)^K)).
 * Returns size_t to avoid float-mantissa truncation past 2^24, and caps
 * at the absolute upper bound M*N. */
size_t cNNZ_estimation(int M, int K, int N, int aNNZ, int bNNZ){
    if (K <= 0 || M <= 0 || N <= 0) return 0;
    size_t est = ((size_t)aNNZ * (size_t)bNNZ) / (size_t)K;
    size_t upper = (size_t)M * (size_t)N;
    return est < upper ? est : upper;
}

int main(){
    int rc = -1;
    FILE *file = NULL, *file1 = NULL, *file2 = NULL, *file3 = NULL;
    int *gInfo = NULL;
    float *aD = NULL, *bD = NULL, *cD = NULL;
    int *aC = NULL, *aR = NULL, *bC = NULL, *bR = NULL, *cC = NULL, *cR = NULL;

    gInfo = (int *)malloc(4*sizeof(int));
    if (gInfo == NULL) { perror("alloc gInfo"); goto cleanup; }

    int number;
    file3 = fopen("matrix_data/GeneralInfo.txt", "r");
    if (file3 == NULL) { perror("Error opening file"); goto cleanup; }

    int idx = 0;
    while (fscanf(file3, "%d", &number) == 1) {
        if (idx >= 4) {
            fprintf(stderr, "GeneralInfo.txt has more than 4 entries\n");
            goto cleanup;
        }
        gInfo[idx++] = number;
    }
    fclose(file3); file3 = NULL;
    if (idx < 4) { fprintf(stderr, "GeneralInfo.txt has fewer than 4 entries\n"); goto cleanup; }

    const int M    = gInfo[0];
    const int K    = gInfo[1];
    const int N    = gInfo[0];   /* TODO: extend GeneralInfo to carry N separately */
    const int aNNZ = gInfo[2];
    const int bNNZ = gInfo[2];   /* TODO: extend GeneralInfo to carry bNNZ separately */
    (void)gInfo[3];              /* rN: reserved for redColIdx wiring */

    if (M <= 0 || K <= 0 || aNNZ <= 0 || bNNZ <= 0) {
        fprintf(stderr, "Invalid dimensions in GeneralInfo.txt\n");
        goto cleanup;
    }

    aD = (float *)malloc(aNNZ*sizeof(float));
    aC = (int *)malloc(aNNZ*sizeof(int));
    aR = (int *)malloc((M+1)*sizeof(int));
    bD = (float *)malloc(bNNZ*sizeof(float));
    bC = (int *)malloc(bNNZ*sizeof(int));
    bR = (int *)malloc((K+1)*sizeof(int));
    if (!aD || !aC || !aR || !bD || !bC || !bR) { perror("alloc CSR"); goto cleanup; }

    file = fopen("matrix_data/CSR_values.txt", "r");
    if (file == NULL) { perror("Error opening file"); goto cleanup; }
    idx = 0;
    float val;
    while (fscanf(file, "%f", &val) == 1) {
        if (idx >= aNNZ) { fprintf(stderr, "CSR_values.txt exceeds aNNZ=%d\n", aNNZ); goto cleanup; }
        aD[idx] = val;
        bD[idx] = val;
        idx++;
    }
    fclose(file); file = NULL;

    file1 = fopen("matrix_data/CSR_colIdx.txt", "r");
    if (file1 == NULL) { perror("Error opening file"); goto cleanup; }
    idx = 0;
    while (fscanf(file1, "%d", &number) == 1) {
        if (idx >= aNNZ) { fprintf(stderr, "CSR_colIdx.txt exceeds aNNZ=%d\n", aNNZ); goto cleanup; }
        aC[idx] = number;
        bC[idx] = number;
        idx++;
    }
    fclose(file1); file1 = NULL;

    file2 = fopen("matrix_data/CSR_rowPtr.txt", "r");
    if (file2 == NULL) { perror("Error opening file"); goto cleanup; }
    idx = 0;
    while (fscanf(file2, "%d", &number) == 1) {
        if (idx >= M+1) { fprintf(stderr, "CSR_rowPtr.txt exceeds M+1=%d\n", M+1); goto cleanup; }
        aR[idx] = number;
        bR[idx] = number;
        idx++;
    }
    fclose(file2); file2 = NULL;

    size_t cNNZ = cNNZ_estimation(M,K,N,aNNZ,bNNZ);
    if (cNNZ == 0) cNNZ = 1;  /* at least allocate one slot */
    printf("Estimated cNNZ = %lu \n", (unsigned long)cNNZ);

    cD = (float *)malloc(cNNZ*sizeof(float));
    cC = (int *)malloc(cNNZ*sizeof(int));
    cR = (int *)malloc((M+1)*sizeof(int));
    if (!cD || !cC || !cR) { perror("alloc output"); goto cleanup; }

    int ip_rc = RowWiseInnerProduct(M, K, N, aD, aC, aR, bD, bC, bR, cD, cC, cR, cNNZ);
    if (ip_rc != 0) {
        fprintf(stderr, "RowWiseInnerProduct: output buffer overflow "
                        "(estimate %lu < actual). Re-run with a larger estimate.\n",
                (unsigned long)cNNZ);
        goto cleanup;
    }

    printf("Actual cNNZ = %d \n", cR[M]);
    rc = 0;

cleanup:
    if (file)  fclose(file);
    if (file1) fclose(file1);
    if (file2) fclose(file2);
    if (file3) fclose(file3);
    free(gInfo);
    free(aD); free(aC); free(aR);
    free(bD); free(bC); free(bR);
    free(cD); free(cC); free(cR);
    return rc;
}
