#ifndef RV_SPARSE_H
#define RV_SPARSE_H

/* We use basic types to avoid newlib dependency in bare-metal compilation */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RV_SPARSE_FP32/RV_SPARSE_FP64 macros allow us to toggle the precision
 * based on ML or Scientific computing needs.
 */
typedef float rvs_float_t; 

/**
 * We use a 32-bit integer (standard 'int' on RV64) for indices. 
 * This is a deliberate architectural decision.
 * A 32-bit index halves the memory bandwidth requirement compared to 64-bit sizes,
 * meaning we can pack twice as many indices into a single RVV vector register.
 */
typedef int rvs_index_t;

/**
 * Compressed Sparse Row (CSR) Format
 * Note on Memory Alignment:
 * For optimal RISC-V Vector (RVV) load/store operations,
 * 'values', 'col_indices', and 'row_ptr' must be aligned to at least the
 * native cache line size (e.g., 64 bytes).
 */
typedef struct {
    rvs_index_t rows;        /**< Number of rows */
    rvs_index_t cols;        /**< Number of columns */
    rvs_index_t nnz;         /**< Number of non-zero elements */
    
    rvs_float_t *values;     /**< Array of non-zero values, length nnz */
    rvs_index_t *col_indices;/**< Array of column indices, length nnz */
    rvs_index_t *row_ptr;    /**< Array of row pointers, length rows + 1 */
} rvs_csr_t;

/**
 * Compressed Sparse Column (CSC) Format
 */
typedef struct {
    rvs_index_t rows;        
    rvs_index_t cols;        
    rvs_index_t nnz;         
    
    rvs_float_t *values;     
    rvs_index_t *row_indices;
    rvs_index_t *col_ptr;    
} rvs_csc_t;

/* --- Memory Management --- */
rvs_csr_t* rvs_csr_create(rvs_index_t rows, rvs_index_t cols, rvs_index_t nnz);
void rvs_csr_free(rvs_csr_t *mat);

/* --- Sparse Matrix-Vector Multiplication (SpMV) --- */
void rvs_csr_spmv_scalar(const rvs_csr_t *A, const rvs_float_t *x, rvs_float_t *y);
void rvs_csr_spmv_rvv(const rvs_csr_t *A, const rvs_float_t *x, rvs_float_t *y);

/* --- Sparse Matrix-Matrix Multiplication (SpMM) --- */
void rvs_csr_spmm_scalar(const rvs_csr_t *A, const rvs_float_t *B, rvs_float_t *C, rvs_index_t B_cols);
void rvs_csr_spmm_rvv(const rvs_csr_t *A, const rvs_float_t *B, rvs_float_t *C, rvs_index_t B_cols);

#ifdef __cplusplus
}
#endif

#endif /* RV_SPARSE_H */
