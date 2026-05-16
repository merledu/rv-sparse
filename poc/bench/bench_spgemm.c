#include "rv_sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)(tv.tv_sec) * 1000.0 + (double)(tv.tv_usec) / 1000.0;
}

// The original implementation from the repository (AxBRowIP.c)
void RowWiseInnerProduct(const int M, const int K, const int N, 
                         float* aD, int* aC, int* aR, 
                         float* bD, int* bC, int* bR, 
                         float* cD, int* cC, int* cR) {
    int row_count = 0;
    for(int row=0; row<M; row++){
        cR[row] = row_count;
        // INNER LOOP ALLOCATION BUG
        float* cD_ = (float *)calloc(K, sizeof(float));
        
        for(int col=aR[row]; col<aR[row+1]; col++){
            float valA = aD[col];
            for(int b_col=bR[aC[col]]; b_col<bR[aC[col]+1]; b_col++){
                float valB = bD[b_col];
                int idx = bC[b_col];
                cD_[idx] += valA * valB;
            }
        }
        
        // O(K) DENSE SCAN BUG
        for (int i=0; i<K; i++){
            if (cD_[i] != 0){
                cC[row_count] = i;
                cD[row_count] = cD_[i];
                row_count++;
            }
        }
        free(cD_);
    }
    cR[M] = row_count;
}

// Helper to generate a random CSR matrix
void generate_random_csr(rv_csr_t* mat, uint32_t M, uint32_t N, float density) {
    // Start with a generous estimate to avoid frequent reallocs
    uint32_t estimated_nnz = (uint32_t)(M * N * density) + 16;
    rv_csr_create(mat, M, N, estimated_nnz);
    
    mat->rowptr[0] = 0;
    uint32_t current_nnz = 0;
    
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            if (((float)rand() / (float)RAND_MAX) < density) {
                if (current_nnz >= mat->nnz) {
                    // Double capacity when exhausted
                    uint32_t new_cap = mat->nnz * 2 + 16;
                    mat->colidx = realloc(mat->colidx, new_cap * sizeof(uint32_t));
                    mat->val    = realloc(mat->val,    new_cap * sizeof(float));
                    mat->nnz    = new_cap;
                }
                mat->colidx[current_nnz] = j;
                mat->val[current_nnz]    = (float)rand() / (float)RAND_MAX;
                current_nnz++;
            }
        }
        mat->rowptr[i+1] = current_nnz;
    }
    mat->nnz = current_nnz; // Set exact nnz
}

int main(int argc, char** argv) {
    int N = 500;
    float density = 0.05f; // 5%

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) density = atof(argv[2]);

    printf("=== SpGEMM Benchmark (N=%d, density=%.2f%%) ===\n", N, density * 100.0f);
    
    srand(42);
    rv_csr_t A;
    generate_random_csr(&A, N, N, density);
    
    printf("Matrix A generated: %u nnz\n", A.nnz);

    // ======================================================
    // Benchmark Original AxBRowIP
    // ======================================================
    
    // The original code guesses NNZ with a heuristic which can underestimate.
    // To give it a fair run without crashing, we allocate the absolute worst
    // case (N*N) so the buffer cannot overflow. This mirrors the OOM risk
    // identified in the proposal for real-world sparse matrices.
    uint32_t cNNZ_est = (uint32_t)N * (uint32_t)N;
    
    float* cD = (float *)malloc(cNNZ_est * sizeof(float));
    int*   cC = (int   *)malloc(cNNZ_est * sizeof(int));
    int*   cR = (int   *)malloc((N + 1)  * sizeof(int));

    double start = get_time_ms();
    
    RowWiseInnerProduct(N, N, N, 
                        A.val, (int*)A.colidx, (int*)A.rowptr,
                        A.val, (int*)A.colidx, (int*)A.rowptr,
                        cD, cC, cR);
                        
    double time_orig = get_time_ms() - start;
    uint32_t nnz_orig = (uint32_t)cR[N];

    free(cD);
    free(cC);
    free(cR);

    // ======================================================
    // Benchmark Two-Phase Gustavson SpGEMM
    // ======================================================
    
    rv_csr_t C;
    
    start = get_time_ms();
    
    rv_spgemm_csr(&A, &A, &C);
    
    double time_new = get_time_ms() - start;
    uint32_t nnz_new = C.nnz;

    // ======================================================
    // Results
    // ======================================================
    
    printf("\nResults for C = A * A\n");
    printf("Original AxBRowIP:   %8.2f ms (NNZ: %u)\n", time_orig, nnz_orig);
    printf("Two-Phase Gustavson: %8.2f ms (NNZ: %u)\n", time_new, nnz_new);
    printf("Speedup:             %8.2fx\n", time_orig / time_new);

    if (nnz_orig != nnz_new) {
        printf("WARNING: Output NNZ mismatch! Orig: %u, New: %u\n", nnz_orig, nnz_new);
    }

    rv_csr_destroy(&A);
    rv_csr_destroy(&C);

    return 0;
}
