#include "rv_sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ASSERT(cond) \
    if (!(cond)) { \
        printf("FAIL: %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    }

#define ASSERT_EQ(a, b) \
    if (fabs((a) - (b)) > 1e-5) { \
        printf("FAIL: %s:%d (Expected %f, got %f)\n", __FILE__, __LINE__, (float)(b), (float)(a)); \
        exit(1); \
    }

int main() {
    printf("Running SpGEMM tests...\n");

    // A = 
    // [ 1  0 ]
    // [ 0  2 ]
    rv_csr_t A;
    rv_csr_create(&A, 2, 2, 2);
    A.rowptr[0] = 0; A.rowptr[1] = 1; A.rowptr[2] = 2;
    A.colidx[0] = 0; A.val[0] = 1.0f;
    A.colidx[1] = 1; A.val[1] = 2.0f;

    // B = 
    // [ 3  4 ]
    // [ 5  0 ]
    rv_csr_t B;
    rv_csr_create(&B, 2, 2, 3);
    B.rowptr[0] = 0; B.rowptr[1] = 2; B.rowptr[2] = 3;
    B.colidx[0] = 0; B.colidx[1] = 1; B.val[0] = 3.0f; B.val[1] = 4.0f;
    B.colidx[2] = 0; B.val[2] = 5.0f;

    // C = A * B
    // [ 3   4 ]
    // [ 10  0 ]
    rv_csr_t C;
    rv_spgemm_csr(&A, &B, &C);

    ASSERT(C.M == 2 && C.N == 2);
    ASSERT(C.nnz == 3);
    
    // Check rowptrs
    ASSERT(C.rowptr[0] == 0);
    ASSERT(C.rowptr[1] == 2);
    ASSERT(C.rowptr[2] == 3);

    // Check colidx and values
    // Row 0
    ASSERT(C.colidx[0] == 0 || C.colidx[0] == 1);
    if (C.colidx[0] == 0) {
        ASSERT_EQ(C.val[0], 3.0f);
        ASSERT(C.colidx[1] == 1);
        ASSERT_EQ(C.val[1], 4.0f);
    } else {
        ASSERT_EQ(C.val[0], 4.0f);
        ASSERT(C.colidx[1] == 0);
        ASSERT_EQ(C.val[1], 3.0f);
    }

    // Row 1
    ASSERT(C.colidx[2] == 0);
    ASSERT_EQ(C.val[2], 10.0f);

    rv_csr_destroy(&A);
    rv_csr_destroy(&B);
    rv_csr_destroy(&C);

    printf("SpGEMM tests passed!\n");
    return 0;
}
