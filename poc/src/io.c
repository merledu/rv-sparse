#include "rv_sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper to count entries per row for COO to CSR conversion
rv_sparse_err_t rv_csr_from_mtx(const char* filepath, rv_csr_t* mat) {
    FILE* f = fopen(filepath, "r");
    if (!f) return RV_SPARSE_ERR_FILE;

    char line[1024];
    uint32_t M = 0, N = 0, nnz = 0;
    int is_symmetric = 0;
    int is_pattern = 0;

    // Read header
    if (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "%%MatrixMarket", 14) != 0) {
            fclose(f);
            return RV_SPARSE_ERR_FORMAT;
        }
        if (strstr(line, "symmetric")) is_symmetric = 1;
        if (strstr(line, "pattern")) is_pattern = 1;
    }

    // Skip comments
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }

    // Parse dimensions
    if (sscanf(line, "%u %u %u", &M, &N, &nnz) != 3) {
        fclose(f);
        return RV_SPARSE_ERR_FORMAT;
    }

    uint32_t true_nnz = nnz;
    if (is_symmetric) {
        // Upper bound, we might not double diagonals
        true_nnz *= 2; 
    }

    // First pass to count nnz per row
    uint32_t* row_counts = (uint32_t*)calloc(M, sizeof(uint32_t));
    
    // We need to store COO temporarily
    uint32_t* coo_row = (uint32_t*)malloc(true_nnz * sizeof(uint32_t));
    uint32_t* coo_col = (uint32_t*)malloc(true_nnz * sizeof(uint32_t));
    float* coo_val = (float*)malloc(true_nnz * sizeof(float));

    uint32_t k = 0;
    for (uint32_t i = 0; i < nnz; i++) {
        if (!fgets(line, sizeof(line), f)) break;
        uint32_t r, c;
        float v = 1.0f;
        if (is_pattern) {
            sscanf(line, "%u %u", &r, &c);
        } else {
            sscanf(line, "%u %u %f", &r, &c, &v);
        }
        r--; c--; // 1-based to 0-based
        
        coo_row[k] = r;
        coo_col[k] = c;
        coo_val[k] = v;
        row_counts[r]++;
        k++;

        if (is_symmetric && r != c) {
            coo_row[k] = c;
            coo_col[k] = r;
            coo_val[k] = v;
            row_counts[c]++;
            k++;
        }
    }
    fclose(f);

    true_nnz = k;

    // Allocate CSR
    rv_csr_create(mat, M, N, true_nnz);

    // Compute rowptr
    mat->rowptr[0] = 0;
    for (uint32_t i = 0; i < M; i++) {
        mat->rowptr[i+1] = mat->rowptr[i] + row_counts[i];
    }

    // Compute colidx and val
    uint32_t* current_row_pos = (uint32_t*)malloc(M * sizeof(uint32_t));
    for (uint32_t i = 0; i < M; i++) {
        current_row_pos[i] = mat->rowptr[i];
    }

    for (uint32_t i = 0; i < true_nnz; i++) {
        uint32_t r = coo_row[i];
        uint32_t pos = current_row_pos[r]++;
        mat->colidx[pos] = coo_col[i];
        mat->val[pos] = coo_val[i];
    }

    free(row_counts);
    free(current_row_pos);
    free(coo_row);
    free(coo_col);
    free(coo_val);

    return RV_SPARSE_SUCCESS;
}
