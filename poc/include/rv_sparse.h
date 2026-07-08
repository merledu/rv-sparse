#ifndef RV_SPARSE_H
#define RV_SPARSE_H

#include <stddef.h>
#include <stdint.h>

// Error codes
typedef enum {
    RV_SPARSE_SUCCESS = 0,
    RV_SPARSE_ERR_ALLOC = -1,
    RV_SPARSE_ERR_FILE = -2,
    RV_SPARSE_ERR_FORMAT = -3,
    RV_SPARSE_ERR_MATH = -4
} rv_sparse_err_t;

// Compressed Sparse Row (CSR) matrix structure
typedef struct {
    uint32_t M;        // Number of rows
    uint32_t N;        // Number of columns
    uint32_t nnz;      // Number of non-zero elements
    uint32_t* rowptr;  // Array of size M+1
    uint32_t* colidx;  // Array of size nnz
    float* val;        // Array of size nnz
} rv_csr_t;

// CSR Management
rv_sparse_err_t rv_csr_create(rv_csr_t* mat, uint32_t M, uint32_t N, uint32_t nnz);
void rv_csr_destroy(rv_csr_t* mat);

// I/O Operations
rv_sparse_err_t rv_csr_from_mtx(const char* filepath, rv_csr_t* mat);

// Mathematical Operations
// SpMV: y = A * x
rv_sparse_err_t rv_spmv_csr(const rv_csr_t* A, const float* x, float* y);

// SpGEMM: C = A * B (Two-Phase Gustavson)
rv_sparse_err_t rv_spgemm_csr(const rv_csr_t* A, const rv_csr_t* B, rv_csr_t* C);

#endif // RV_SPARSE_H
