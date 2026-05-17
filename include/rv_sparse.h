#ifndef RV_SPARSE_H
#define RV_SPARSE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
	int num_rows;
	int num_cols;
	int nnz;
	float *values;
	int *col_idx;
	int *row_ptr;

} CSRMatrix_f32;

typedef struct
{
	int num_rows;
	int num_cols;
	int nnz;
	double *values;
	int *col_idx;
	int *row_ptr;
} CSRMatrix_f64;

CSRMatrix_f32 *csr_alloc_f32(int rows, int cols, int nnz);
void csr_free_f32(CSRMatrix_f32 *A);
CSRMatrix_f64 *csr_alloc_f64(int rows, int cols, int nnz);
void csr_free_f64(CSRMatrix_f64 *A);

void spmv_csr_scalar_f32(const CSRMatrix_f32 *A, const float *x, float *y);
void spmv_csr_scalar_f64(const CSRMatrix_f64 *A, const double *x, double *y);
void spmv_csr_rvv_f32(const CSRMatrix_f32 *A, const float *x, float *y);
void spmv_csr_rvv_f64(const CSRMatrix_f64 *A, const double *x, double *y);
void spmv_csr_scalar_f32_counted(const CSRMatrix_f32 *A, const float *x,
								 float *y, uint64_t *ops);
void spmv_csr_rvv_f32_counted(const CSRMatrix_f32 *A, const float *x,
							  float *y, uint64_t *ops, uint64_t *elems);

void spmm_csr_scalar_f32(const CSRMatrix_f32 *A, const float *B, int B_cols, float *C);
void spmm_csr_scalar_f64(const CSRMatrix_f64 *A, const double *B, int B_cols, double *C);
void spmm_csr_rvv_f32(const CSRMatrix_f32 *A, const float *B, int B_cols, float *C);
void spmm_csr_rvv_f64(const CSRMatrix_f64 *A, const double *B, int B_cols, double *C);

CSRMatrix_f32 *generate_random_csr_f32(int rows, int cols, float density, unsigned int seed);

#endif
