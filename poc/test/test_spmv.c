#include "rv_sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ASSERT_EQ(a, b) \
    if (fabs((a) - (b)) > 1e-5) { \
        printf("FAIL: %s:%d (Expected %f, got %f)\n", __FILE__, __LINE__, (float)(b), (float)(a)); \
        exit(1); \
    }

int main() {
    printf("Running SpMV tests...\n");

    // Construct a simple 3x3 CSR matrix
    // [ 1  2  0 ]
    // [ 0  3  4 ]
    // [ 5  0  6 ]
    rv_csr_t A;
    rv_csr_create(&A, 3, 3, 6);
    
    A.rowptr[0] = 0; A.rowptr[1] = 2; A.rowptr[2] = 4; A.rowptr[3] = 6;
    
    A.colidx[0] = 0; A.colidx[1] = 1; A.val[0] = 1.0f; A.val[1] = 2.0f;
    A.colidx[2] = 1; A.colidx[3] = 2; A.val[2] = 3.0f; A.val[3] = 4.0f;
    A.colidx[4] = 0; A.colidx[5] = 2; A.val[4] = 5.0f; A.val[5] = 6.0f;

    // x = [1, 1, 1]^T
    float x1[] = {1.0f, 1.0f, 1.0f};
    float y1[3] = {0};
    rv_spmv_csr(&A, x1, y1);
    ASSERT_EQ(y1[0], 3.0f);
    ASSERT_EQ(y1[1], 7.0f);
    ASSERT_EQ(y1[2], 11.0f);

    // x = [2, -1, 0]^T
    float x2[] = {2.0f, -1.0f, 0.0f};
    float y2[3] = {0};
    rv_spmv_csr(&A, x2, y2);
    ASSERT_EQ(y2[0], 0.0f);
    ASSERT_EQ(y2[1], -3.0f);
    ASSERT_EQ(y2[2], 10.0f);

    rv_csr_destroy(&A);
    printf("SpMV tests passed!\n");
    return 0;
}
