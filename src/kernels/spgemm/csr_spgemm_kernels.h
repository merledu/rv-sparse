/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Internal CSR SpGEMM wrapper layer.
 * Bridges the public API with raw pointer-based backend kernels.
 */

#ifndef RVSP_CSR_SPGEMM_KERNELS_H
#define RVSP_CSR_SPGEMM_KERNELS_H

#include <stdint.h>
#include "rv_sparse.h"

/*
 * Wrapper layer:
 * Uses rvsp_csr_matrix_t structs.
 * Called by the public dispatcher.
 */

// Initial kernels

rvsp_status_t rvsp_spgemm_csr_scalar_i8(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C);

rvsp_status_t rvsp_spgemm_csr_scalar_f32(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C);

// Optimizations : unroll, fused , vectorized, etc ....
rvsp_status_t rvsp_spgemm_csr_scalar_unroll4_f32(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C);

/*
 * Raw kernel layer:
 * Uses plain pointers and dimensions.
 */

rvsp_status_t rvsp_spgemm_csr_scalar_i8_raw(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *restrict a_row_ptr,
    const int32_t *restrict a_col_idx,
    const int8_t *restrict a_values,
    const int32_t *restrict b_row_ptr,
    const int32_t *restrict b_col_idx,
    const int8_t *restrict b_values,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    int32_t **c_values_out,
    int32_t *c_nnz_out);

rvsp_status_t rvsp_spgemm_csr_scalar_f32_raw(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *restrict a_row_ptr,
    const int32_t *restrict a_col_idx,
    const float *restrict a_values,
    const int32_t *restrict b_row_ptr,
    const int32_t *restrict b_col_idx,
    const float *restrict b_values,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    float **c_values_out,
    int32_t *c_nnz_out);

rvsp_status_t rvsp_spgemm_csr_scalar_unroll4_f32_raw(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *restrict a_row_ptr,
    const int32_t *restrict a_col_idx,
    const float *restrict a_values,
    const int32_t *restrict b_row_ptr,
    const int32_t *restrict b_col_idx,
    const float *restrict b_values,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    float **c_values_out,
    int32_t *c_nnz_out);

// This is the layer we optimize and port to RISC-V/RVV.
rvsp_status_t rvsp_accumulate_row_i8_rvv_indexed_fast(
    int8_t a_val,
    int32_t b_nnz,
    const int32_t *b_col_idx,
    const int8_t *b_values,
    int32_t *acc);

rvsp_status_t rvsp_spgemm_csr_rvv_i8_indexed_marked_raw(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *a_row_ptr,
    const int32_t *a_col_idx,
    const int8_t *a_values,
    const int32_t *b_row_ptr,
    const int32_t *b_col_idx,
    const int8_t *b_values,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    int32_t **c_values_out,
    int32_t *c_nnz_out);

rvsp_status_t rvsp_spgemm_csr_rvv_i8_indexed_marked(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C);

rvsp_status_t rvsp_accumulate_row_f32_rvv_indexed_fast(
    float a_val,
    int32_t b_nnz,
    const int32_t *b_col_idx,
    const float *b_values,
    float *acc);

rvsp_status_t rvsp_spgemm_csr_rvv_f32_indexed_marked_raw(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *a_row_ptr,
    const int32_t *a_col_idx,
    const float *a_values,
    const int32_t *b_row_ptr,
    const int32_t *b_col_idx,
    const float *b_values,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    float **c_values_out,
    int32_t *c_nnz_out);

rvsp_status_t rvsp_spgemm_csr_rvv_f32_indexed_marked(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C);

#endif /*RVSP_CSR_SPGEMM_KERNELS_H*/