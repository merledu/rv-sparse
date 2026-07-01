#include <stdint.h>
#include <stddef.h>

#include "csr_spgemm_kernels.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#define EXP_HAVE_RVV_INTRINSICS 1
#endif

rvsp_status_t rvsp_accumulate_row_f32_rvv_indexed_fast(float a_val, int32_t b_nnz,
                                                      const int32_t *b_col_idx,
                                                      const float *b_values,
                                                      float *acc)
{
    if (b_nnz < 0 || b_col_idx == NULL || b_values == NULL || acc == NULL)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

#if defined(EXP_HAVE_RVV_INTRINSICS)
    int32_t p = 0;

    while (p < b_nnz)
    {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(b_nnz - p));

        vfloat32m4_t vb = __riscv_vle32_v_f32m4(&b_values[p], vl);
        vfloat32m4_t vprod = __riscv_vfmul_vf_f32m4(vb, a_val, vl);

        vint32m4_t vidx_i32 = __riscv_vle32_v_i32m4(&b_col_idx[p], vl);
        vint32m4_t voff_i32 = __riscv_vsll_vx_i32m4(vidx_i32, 2, vl);
        vuint32m4_t voff_u32 = __riscv_vreinterpret_v_i32m4_u32m4(voff_i32);

        vfloat32m4_t vacc = __riscv_vluxei32_v_f32m4(acc, voff_u32, vl);
        vacc = __riscv_vfadd_vv_f32m4(vacc, vprod, vl);
        __riscv_vsuxei32_v_f32m4(acc, voff_u32, vacc, vl);

        p += (int32_t)vl;
    }
#else
    for (int32_t p = 0; p < b_nnz; ++p)
    {
        int32_t col = b_col_idx[p];
        acc[col] += a_val * b_values[p];
    }
#endif

    return RVSP_SUCCESS;
}
