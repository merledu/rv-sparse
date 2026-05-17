//test_rvv_ops.c
#include <stdio.h>
#include <math.h>
#include <riscv_vector.h>

// Test 1: contiguous load + store
static int test_load_store() {
    float a[8] = {1,2,3,4,5,6,7,8};
    float b[8] = {0};
    size_t vl = __riscv_vsetvl_e32m1(8);
    vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
    __riscv_vse32_v_f32m1(b, va, vl);
    for (int i = 0; i < 8; i++)
        if (fabsf(a[i]-b[i]) > 1e-6f) return 0;
    return 1;
}

// Test 2: gather (vloxei32) — THE critical operation
static int test_gather() {
    float x[10] = {0,10,20,30,40,50,60,70,80,90};
    uint32_t idx[4] = {7, 2, 9, 0};          // element indices
    uint32_t byte_offsets[4];
    for (int i = 0; i < 4; i++)
        byte_offsets[i] = idx[i] * sizeof(float); // MUST be byte offsets

    size_t vl = __riscv_vsetvl_e32m1(4);
    vuint32m1_t voffsets = __riscv_vle32_v_u32m1(byte_offsets, vl);
    vfloat32m1_t vgathered = __riscv_vloxei32_v_f32m1(x, voffsets, vl);

    float result[4];
    __riscv_vse32_v_f32m1(result, vgathered, vl);

    float expected[4] = {70,20,90,0};
    for (int i = 0; i < 4; i++)
        if (fabsf(result[i]-expected[i]) > 1e-6f) return 0;
    return 1;
}


static int test_fmacc() {
    float a[4] = {1,2,3,4};
    float b[4] = {2,2,2,2};
    size_t vl = __riscv_vsetvl_e32m1(4);
    vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
    vfloat32m1_t vc = __riscv_vfmul_vv_f32m1(va, vb, vl); // {2,4,6,8}
    float result[4];
    __riscv_vse32_v_f32m1(result, vc, vl);
    return fabsf(result[0]-2)+fabsf(result[3]-8) < 1e-5f;
}


static int test_reduction() {
    float a[4] = {1,2,3,4};
    size_t vl = __riscv_vsetvl_e32m1(4);
    vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vsum  = __riscv_vfredosum_vs_f32m1_f32m1(va, vzero, vl);
    float result = __riscv_vfmv_f_s_f32m1_f32(vsum);
    return fabsf(result - 10.0f) < 1e-5f;
}

int main() {
    printf("load_store:  %s\n", test_load_store() ? "PASS" : "FAIL");
    printf("gather:      %s\n", test_gather()     ? "PASS" : "FAIL");
    printf("fmacc:       %s\n", test_fmacc()      ? "PASS" : "FAIL");
    printf("reduction:   %s\n", test_reduction()  ? "PASS" : "FAIL");
    return 0;
}
