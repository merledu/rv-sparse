/*
 * spgemm_csr_f32_otf.c
 *
 * End-to-end SpGEMM example using:
 *   - genmat: generates a sparse CSR matrix on-the-fly (no file needed)
 *   - mtx parser: loads a real matrix from a Matrix Market file
 *
 * Demonstrates two paths into rvsp_spgemm_csr:
 *   1. Generated matrix A (via genmat)
 *   2. Parsed matrix B (via assemble_csr_matrix from ash85.mtx)
 *
 * Build: handled by Makefile (auto-discovered in examples/)
 * Run:   ./bin/examples/spgemm_csr_f32_otf
 */

#include "genmat.h"
#include "mtx_to_csr_formatter.h"
#include "rv_sparse.h"

#include <stdio.h>
#include <stdlib.h>

static void print_csr(const rvsp_csr_matrix_t *M) {
  const float *values = (const float *)M->values;
  printf("  %dx%d, nnz=%d\n", M->rows, M->cols, M->nnz);
  for (int32_t i = 0; i < M->rows; i++) {
    printf("  row %d:", i);
    for (int32_t k = M->row_ptr[i]; k < M->row_ptr[i + 1]; k++) {
      printf(" (%d, %.2f)", M->col_idx[k], values[k]);
    }
    printf("\n");
  }
}

int main(void) {
  rvsp_status_t status;

  // -----------------------------------------------------------------------
  // Matrix A: generated on-the-fly with genmat
  // -----------------------------------------------------------------------
  printf("=== Matrix A (generated) ===\n");

  genmat_params_t p = genmat_default_params(85, 85);
  p.density = 0.05;
  p.cv = 0.3;
  p.random_seed = 42;

  csr_matrix_t gen = genmat_generate_csr(p);
  if (gen.nrows < 0) {
    fprintf(stderr, "genmat_generate_csr failed\n");
    return EXIT_FAILURE;
  }
  printf("  generated %dx%d, nnz=%lld\n", gen.nrows, gen.ncols, gen.nnz);

  rvsp_csr_matrix_t A;
  status =
      rvsp_csr_create(&A, gen.nrows, gen.ncols, (int32_t)gen.nnz, gen.row_ptr,
                      gen.col_idx, gen.values, RVSP_DTYPE_FP32);
  if (status != RVSP_SUCCESS) {
    printf("Failed to create A: %s\n", rvsp_status_to_string(status));
    genmat_free_csr(&gen);
    return EXIT_FAILURE;
  }

  // -----------------------------------------------------------------------
  // Matrix B: parsed from ash85.mtx
  // -----------------------------------------------------------------------
  printf("\n=== Matrix B (parsed from ash85.mtx) ===\n");

  struct CSR parsed = assemble_csr_matrix("examples/ash85.mtx");
  if (parsed.row_ptr == NULL || parsed.col_ind == NULL || parsed.val == NULL) {
    fprintf(stderr, "assemble_csr_matrix failed\n");
    genmat_free_csr(&gen);
    rvsp_csr_destroy(&A);
    return EXIT_FAILURE;
  }

  int M_rows = (int)vector_size(parsed.row_ptr) - 1;
  int M_nnz = (int)vector_size(parsed.col_ind);
  printf("  parsed %dx%d, nnz=%d\n", M_rows, M_rows, M_nnz);

  rvsp_csr_matrix_t B;
  status = rvsp_csr_create(&B, M_rows, M_rows, M_nnz,
                           (int32_t *)parsed.row_ptr->data,
                           (int32_t *)parsed.col_ind->data,
                           (float *)parsed.val->data, RVSP_DTYPE_FP32);
  if (status != RVSP_SUCCESS) {
    printf("Failed to create B: %s\n", rvsp_status_to_string(status));
    genmat_free_csr(&gen);
    rvsp_csr_destroy(&A);
    vector_free(parsed.row_ptr);
    vector_free(parsed.col_ind);
    vector_free(parsed.val);
    return EXIT_FAILURE;
  }

  // -----------------------------------------------------------------------
  // SpGEMM: C = A * B  (both must be same shape, using 85x85 here)
  // -----------------------------------------------------------------------
  printf("\n=== SpGEMM: C = A x B ===\n");

  rvsp_csr_matrix_t C = {0};
  rvsp_spgemm_options_t options = {
      .backend = RVSP_BACKEND_SCALAR,
      .input_dtype = RVSP_DTYPE_FP32,
      .output_dtype = RVSP_DTYPE_FP32,
      .sort_output_indices = 1,
  };

  status = rvsp_spgemm_csr(&A, &B, &C, &options);
  if (status != RVSP_SUCCESS) {
    printf("SpGEMM failed: %s\n", rvsp_status_to_string(status));
  } else {
    printf("  result C: %dx%d, nnz=%d\n", C.rows, C.cols, C.nnz);
  }

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------
  genmat_free_csr(&gen);
  vector_free(parsed.row_ptr);
  vector_free(parsed.col_ind);
  vector_free(parsed.val);
  rvsp_csr_destroy(&A);
  rvsp_csr_destroy(&B);
  rvsp_csr_destroy(&C);

  return EXIT_SUCCESS;
}
