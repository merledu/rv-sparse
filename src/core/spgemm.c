/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of rv-sparse.
 *
 * rv-sparse is an experimental sparse linear algebra library focused on
 * portable sparse kernels and future RISC-V optimization.
 *
 * This source file implements part of the internal backend infrastructure
 * used by the public API.
 */

/*Note:
 * this is a basic dispacher, inspired in blas, from her the library decided what kernel use
 * in revision
 * todo things :  structure for add new kernels easily..
 */

#include "rv_sparse.h"
#include "../kernels/spgemm/csr_spgemm_kernels.h"

rvsp_status_t rvsp_spgemm_csr(const rvsp_csr_matrix_t *A, const rvsp_csr_matrix_t *B,
                              rvsp_csr_matrix_t *C,
                              const rvsp_spgemm_options_t *options)
{
    rvsp_backend_t backend = RVSP_BACKEND_SCALAR;
    rvsp_dtype_t input_dtype = RVSP_DTYPE_FP32;
    rvsp_dtype_t output_dtype = RVSP_DTYPE_FP32;

    if (!A || !B || !C)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (options)
    {
        backend = options->backend;
        input_dtype = options->input_dtype;
        output_dtype = options->output_dtype;
    }

    if (backend == RVSP_BACKEND_SCALAR &&
        input_dtype == RVSP_DTYPE_FP32 &&
        output_dtype == RVSP_DTYPE_FP32)
    {
        return rvsp_spgemm_csr_scalar_f32(A, B, C);
    }

    if (backend == RVSP_BACKEND_SCALAR_UNROLL4 &&
        input_dtype == RVSP_DTYPE_FP32 &&
        output_dtype == RVSP_DTYPE_FP32)
    {
        return rvsp_spgemm_csr_scalar_unroll4_f32(A, B, C);
    }

    // default error
    return RVSP_ERROR_UNSUPPORTED_BACKEND;
}
