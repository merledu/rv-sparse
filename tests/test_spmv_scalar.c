#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rv_sparse.h"

int main() {
    printf("[TEST] Running SpMV Scalar Verification...\n");
    
    // Create a 3x3 identity matrix in CSR format manually
    rvs_csr_t A;
    A.rows = 3;
    A.cols = 3;
    A.nnz = 3;
    
    rvs_float_t values[] = {1.0, 1.0, 1.0};
    rvs_index_t col_indices[] = {0, 1, 2};
    rvs_index_t row_ptr[] = {0, 1, 2, 3};
    
    A.values = values;
    A.col_indices = col_indices;
    A.row_ptr = row_ptr;
    
    // Input vector
    rvs_float_t x[] = {2.0, 4.0, 6.0};
    
    // Output vector
    rvs_float_t y[] = {0.0, 0.0, 0.0};
    
    // Execute scalar SpMV
    rvs_csr_spmv_scalar(&A, x, y);
    
    // Verify results
    int passed = 1;
    if (fabs(y[0] - 2.0) > 1e-6) passed = 0;
    if (fabs(y[1] - 4.0) > 1e-6) passed = 0;
    if (fabs(y[2] - 6.0) > 1e-6) passed = 0;
    
    if (passed) {
        printf("[TEST] SUCCESS! SpMV Scalar matches ground truth.\n");
        return 0;
    } else {
        printf("[TEST] FAILED! Output: %f, %f, %f\n", y[0], y[1], y[2]);
        return 1;
    }
}
