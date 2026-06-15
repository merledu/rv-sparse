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
rvsp_status_t rvsp_spgemm_csr_scalar_f32(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C);

rvsp_status_t rvsp_spgemm_csr_scalar_unroll4_f32(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C);

/*
 * Raw kernel layer:
 * Uses plain pointers and dimensions.
 * This is the layer we optimize and port to RISC-V/RVV.
 */
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

#endif /*RVSP_CSR_SPGEMM_KERNELS_H*/