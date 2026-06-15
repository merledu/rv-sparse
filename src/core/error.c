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

#include "rv_sparse.h"

const char *rvsp_get_version(void)
{
    return "0.1.0";
}

const char *rvsp_status_to_string(rvsp_status_t status)
{
    switch (status)
    {
    case RVSP_SUCCESS:
        return "success";
    case RVSP_ERROR_NULL_POINTER:
        return "null pointer";
    case RVSP_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case RVSP_ERROR_INVALID_CSR:
        return "invalid CSR matrix";
    case RVSP_ERROR_UNSUPPORTED_DTYPE:
        return "unsupported data type";
    case RVSP_ERROR_UNSUPPORTED_BACKEND:
        return "unsupported backend";
    case RVSP_ERROR_ALLOCATION_FAILED:
        return "allocation failed";
    default:
        return "unknown error";
    }
}
