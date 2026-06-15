/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * Public API for the rv-sparse library.
 */

#ifndef RV_SPARSE_H
#define RV_SPARSE_H

#include "rv_sparse_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define RVSP_VERSION_MAJOR 0
#define RVSP_VERSION_MINOR 1
#define RVSP_VERSION_PATCH 0

    const char *rvsp_get_version(void);

    const char *rvsp_status_to_string(rvsp_status_t status);

    /*
     * CSR matrix lifecycle.
     *
     * rvsp_csr_create() does not take ownership of user-provided arrays.
     * rvsp_spgemm_csr() may allocate output arrays for C.
     * rvsp_csr_destroy() frees internal arrays only when owns_data is set.
     */
    rvsp_status_t rvsp_csr_create(
        rvsp_csr_matrix_t *A,
        int32_t rows,
        int32_t cols,
        int32_t nnz,
        int32_t *row_ptr,
        int32_t *col_idx,
        void *values,
        rvsp_dtype_t dtype);

    rvsp_status_t rvsp_csr_validate(const rvsp_csr_matrix_t *A);

    void rvsp_csr_destroy(rvsp_csr_matrix_t *A);

    /*
     * Sparse matrix-matrix multiplication:
     *
     *     C = A * B
     *
     * Current implementation:
     *     FP32 x FP32 -> FP32 scalar backend.
     *
     * Planned next target:
     *     INT8 x INT8 -> INT32
     */
    rvsp_status_t rvsp_spgemm_csr(
        const rvsp_csr_matrix_t *A,
        const rvsp_csr_matrix_t *B,
        rvsp_csr_matrix_t *C,
        const rvsp_spgemm_options_t *options);

#ifdef __cplusplus
}
#endif

#endif /* RV_SPARSE_H */