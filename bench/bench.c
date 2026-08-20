/*
 * SpGEMM benchmark harness.
 *
 * Builds A and B, validates the selected kernel, runs warmups and timed
 * iterations, and emits one CSV row per run. Setup and validation are untimed.
 *
 * Usage:
 *   ./bench --kernel NAME \
 *     (--gen R C DENSITY SEED | --mtx A.mtx B.mtx | --mtx-sq M.mtx) \
 *     [--runs N] [--warmup W] [--label TAG] [--header] \
 *     [--arm ARM] [--build TAG] [--march FLAGS] [--cflags FLAGS] \
 *     [--cc-version VER]
 *
 * Input modes:
 *   --gen R C DENSITY SEED   Generate synthetic CSR matrices.
 *   --mtx A.mtx B.mtx       Load two Matrix Market matrices.
 *   --mtx-sq M.mtx          Benchmark M * M.
 *
 * Build metadata is supplied by the caller and recorded in the CSV so
 * results from different binaries can share the same dataset.
 */

#define _POSIX_C_SOURCE 199309L  /* clock_gettime / CLOCK_MONOTONIC */
#define _GNU_SOURCE              /* syscall() for perf_event_open */

#include "rv_sparse.h"
#include "rvsp_spgemm.h"
#include "genmat.h"
#include "mtx_to_csr_formatter.h"
#include "vec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/*
 * Optional hardware counters. When enabled, counters cover only the
 * kernel call and are left blank when unavailable.
 */
#ifdef USE_PERF
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <errno.h>

static int perf_fd_cycles = -1;
static int perf_fd_insns  = -1;

static int perf_open_one(uint32_t type, uint64_t config, int group_fd) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type           = type;
    pe.size           = sizeof(pe);
    pe.config         = config;
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;

    long fd = syscall(__NR_perf_event_open, &pe, 0, -1, group_fd, 0);
    return (int)fd;
}

static void perf_init(void) {
    perf_fd_cycles =
        perf_open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1);
    perf_fd_insns =
        perf_open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1);

    if (perf_fd_cycles < 0 || perf_fd_insns < 0)
        fprintf(stderr,
                "warning: perf counters unavailable (%s); "
                "cycles/instructions will be blank\n",
                strerror(errno));
}

static void perf_start(void) {
    if (perf_fd_cycles >= 0) {
        ioctl(perf_fd_cycles, PERF_EVENT_IOC_RESET, 0);
        ioctl(perf_fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
    }
    if (perf_fd_insns >= 0) {
        ioctl(perf_fd_insns, PERF_EVENT_IOC_RESET, 0);
        ioctl(perf_fd_insns, PERF_EVENT_IOC_ENABLE, 0);
    }
}

static void perf_stop(long long *cycles, long long *insns) {
    *cycles = -1;
    *insns = -1;

    if (perf_fd_cycles >= 0) {
        ioctl(perf_fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
        if (read(perf_fd_cycles, cycles, sizeof(*cycles)) != sizeof(*cycles))
            *cycles = -1;
    }

    if (perf_fd_insns >= 0) {
        ioctl(perf_fd_insns, PERF_EVENT_IOC_DISABLE, 0);
        if (read(perf_fd_insns, insns, sizeof(*insns)) != sizeof(*insns))
            *insns = -1;
    }
}

static void perf_close(void) {
    if (perf_fd_cycles >= 0)
        close(perf_fd_cycles);
    if (perf_fd_insns >= 0)
        close(perf_fd_insns);
}
#else
static void perf_init(void) {}
static void perf_start(void) {}
static void perf_stop(long long *cycles, long long *insns) {
    *cycles = -1;
    *insns = -1;
}
static void perf_close(void) {}
#endif

/* Uniform harness signature; adapters hide kernel-specific arguments. */
typedef rvsp_status_t (*spgemm_fn)(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C,
    void *workspace,
    size_t workspace_bytes,
    int64_t *op_counts_out);

/* Kernel declarations used by the benchmark registry. */
/*
 * Adapt the raw kernel API to rvsp_csr_matrix_t.
 * Workspace is supplied by the harness and reused across runs.
 */
#define KERNEL_WRAP(wrapper, kernel)                                      \
    static rvsp_status_t wrapper(                                    \
        const rvsp_csr_matrix_t *A,                                  \
        const rvsp_csr_matrix_t *B,                                  \
        rvsp_csr_matrix_t *C,                                        \
        void *ws, size_t wsb, int64_t *ops) {                         \
        int32_t *c_row_ptr = NULL, *c_col_idx = NULL, c_nnz = 0;      \
        float *c_values = NULL;                                      \
        rvsp_status_t st = kernel(                                   \
            A->rows, A->cols, B->cols,                               \
            A->row_ptr, A->col_idx, (const float *)A->values,        \
            B->row_ptr, B->col_idx, (const float *)B->values,        \
            ws, wsb,                                                 \
            &c_row_ptr, &c_col_idx, &c_values, &c_nnz, ops);          \
        if (st != RVSP_SUCCESS)                                      \
            return st;                                               \
        C->rows = A->rows;                                           \
        C->cols = B->cols;                                           \
        C->nnz = c_nnz;                                              \
        C->row_ptr = c_row_ptr;                                      \
        C->col_idx = c_col_idx;                                      \
        C->values = c_values;                                        \
        C->dtype = RVSP_DTYPE_FP32;                                  \
        C->format = RVSP_FORMAT_CSR;                                 \
        C->owns_data = 1;                                            \
        return RVSP_SUCCESS;                                         \
    }

KERNEL_WRAP(scalar_f32_w, rvsp_spgemm_scalar_f32)

#if defined(__riscv_vector)
KERNEL_WRAP(rvv_f32_w,      rvsp_spgemm_rvv_f32)
KERNEL_WRAP(contig_f32_w,   rvsp_spgemm_contig_f32)
KERNEL_WRAP(adaptive_f32_w, rvsp_spgemm_adaptive_f32)
#endif

typedef struct {
    const char   *name;
    spgemm_fn     fn;
    rvsp_dtype_t  dtype;
    const char   *dtype_str;
    spgemm_fn     ref;
    const char   *ref_name;
} kernel_entry_t;

/* Each current kernel is validated against the matching scalar reference. */
static const kernel_entry_t KERNELS[] = {
    { "scalar_f32", scalar_f32_w, RVSP_DTYPE_FP32, "f32",
      scalar_f32_w, "scalar_f32" },

#if defined(__riscv_vector)
    { "rvv_f32", rvv_f32_w, RVSP_DTYPE_FP32, "f32",
      scalar_f32_w, "scalar_f32" },
    { "contig_f32", contig_f32_w, RVSP_DTYPE_FP32, "f32",
      scalar_f32_w, "scalar_f32" },
    { "adaptive_f32", adaptive_f32_w, RVSP_DTYPE_FP32, "f32",
      scalar_f32_w, "scalar_f32" },
#endif
};

static const int N_KERNELS =
    (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));

static const kernel_entry_t *find_kernel(const char *name) {
    for (int i = 0; i < N_KERNELS; i++)
        if (strcmp(KERNELS[i].name, name) == 0)
            return &KERNELS[i];
    return NULL;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Compute the intermediate products for A*B and record Op_i per output row. */
static double intermediate_products(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    int64_t *op_counts) {

    double prods = 0.0;

    for (int32_t i = 0; i < A->rows; i++) {
        int64_t row_ops = 0;

        for (int32_t p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            int32_t k = A->col_idx[p];  /* A column = B row */

            if (k >= 0 && k < B->rows)
                row_ops +=
                    (int64_t)(B->row_ptr[k + 1] - B->row_ptr[k]);
        }

        if (op_counts)
            op_counts[i] = row_ops;

        prods += (double)row_ops;
    }

    return prods;
}

/* Mean, maximum, and population variance of Op_i over the output rows. */
static void op_stats(
    const int64_t *op,
    int32_t n,
    double *mean_out,
    double *max_out,
    double *var_out) {

    *mean_out = 0.0;
    *max_out = 0.0;
    *var_out = 0.0;

    if (!op || n <= 0)
        return;

    double sum = 0.0;
    int64_t mx = 0;

    for (int32_t i = 0; i < n; i++) {
        sum += (double)op[i];
        if (op[i] > mx)
            mx = op[i];
    }

    const double mean = sum / (double)n;

    double acc = 0.0;
    for (int32_t i = 0; i < n; i++) {
        const double d = (double)op[i] - mean;
        acc += d * d;
    }

    *mean_out = mean;
    *max_out = (double)mx;
    *var_out = acc / (double)n;
}

/*
 * Compare CSR outputs independent of column ordering within each row.
 * Values use dtype-specific tolerances.
 */
typedef struct {
    int32_t col;
    int32_t pos;
} colref_t;

static int cmp_colref(const void *a, const void *b) {
    int32_t ca = ((const colref_t *)a)->col;
    int32_t cb = ((const colref_t *)b)->col;
    return (ca > cb) - (ca < cb);
}

static int vals_close(
    const void *xv,
    const void *yv,
    int32_t px,
    int32_t py,
    rvsp_dtype_t dtype) {

    if (dtype == RVSP_DTYPE_FP32) {
        float a = ((const float *)xv)[px];
        float b = ((const float *)yv)[py];
        return fabsf(a - b) <= 1e-5f + 1e-4f * fabsf(b);
    } else if (dtype == RVSP_DTYPE_FP64) {
        double a = ((const double *)xv)[px];
        double b = ((const double *)yv)[py];
        return fabs(a - b) <= 1e-12 + 1e-9 * fabs(b);
    } else {
        int32_t a = ((const int32_t *)xv)[px];
        int32_t b = ((const int32_t *)yv)[py];
        return a == b;
    }
}

static int csr_equal(
    const rvsp_csr_matrix_t *X,
    const rvsp_csr_matrix_t *Y,
    rvsp_dtype_t dtype) {

    if (X->rows != Y->rows ||
        X->cols != Y->cols ||
        X->nnz != Y->nnz)
        return 0;

    for (int32_t i = 0; i <= X->rows; i++)
        if (X->row_ptr[i] != Y->row_ptr[i])
            return 0;

    int32_t maxlen = 0;

    for (int32_t i = 0; i < X->rows; i++) {
        int32_t len = X->row_ptr[i + 1] - X->row_ptr[i];
        if (len > maxlen)
            maxlen = len;
    }

    if (maxlen == 0)
        return 1;

    colref_t *rx = malloc((size_t)maxlen * sizeof(colref_t));
    colref_t *ry = malloc((size_t)maxlen * sizeof(colref_t));

    if (!rx || !ry) {
        free(rx);
        free(ry);
        return 0;
    }

    int ok = 1;

    for (int32_t i = 0; i < X->rows && ok; i++) {
        int32_t sx = X->row_ptr[i];
        int32_t sy = Y->row_ptr[i];
        int32_t len = X->row_ptr[i + 1] - sx;

        for (int32_t t = 0; t < len; t++) {
            rx[t].col = X->col_idx[sx + t];
            rx[t].pos = sx + t;
            ry[t].col = Y->col_idx[sy + t];
            ry[t].pos = sy + t;
        }

        qsort(rx, (size_t)len, sizeof(colref_t), cmp_colref);
        qsort(ry, (size_t)len, sizeof(colref_t), cmp_colref);

        for (int32_t t = 0; t < len; t++) {
            if (rx[t].col != ry[t].col ||
                !vals_close(X->values, Y->values,
                             rx[t].pos, ry[t].pos, dtype)) {
                ok = 0;
                break;
            }
        }
    }

    free(rx);
    free(ry);
    return ok;
}

/* Load inputs into typed CSR storage owned by the harness. */
typedef struct {
    int32_t *row_ptr;
    int32_t *col_idx;
    void    *values;
    int      free_via_genmat;
} raw_csr_t;

/* Convert generated CSR values to the selected kernel dtype. */
static int build_from_genmat(
    csr_matrix_t *g,
    rvsp_dtype_t dtype,
    rvsp_csr_matrix_t *out,
    raw_csr_t *raw) {

    int32_t rows = (int32_t)g->nrows;
    int32_t cols = (int32_t)g->ncols;
    int32_t nnz  = (int32_t)g->nnz;

    raw->row_ptr = malloc((size_t)(rows + 1) * sizeof(int32_t));
    raw->col_idx = malloc((size_t)nnz * sizeof(int32_t));

    if (!raw->row_ptr || !raw->col_idx)
        return -1;

    for (int32_t i = 0; i <= rows; i++)
        raw->row_ptr[i] = (int32_t)g->row_ptr[i];

    for (int32_t p = 0; p < nnz; p++)
        raw->col_idx[p] = (int32_t)g->col_idx[p];

    if (dtype == RVSP_DTYPE_FP32) {
        float *v = malloc((size_t)nnz * sizeof(float));
        if (!v)
            return -1;

        for (int32_t p = 0; p < nnz; p++)
            v[p] = (float)g->values[p];

        raw->values = v;
    } else if (dtype == RVSP_DTYPE_FP64) {
        double *v = malloc((size_t)nnz * sizeof(double));
        if (!v)
            return -1;

        for (int32_t p = 0; p < nnz; p++)
            v[p] = g->values[p];

        raw->values = v;
    } else {
        int8_t *v = malloc((size_t)nnz * sizeof(int8_t));
        if (!v)
            return -1;

        for (int32_t p = 0; p < nnz; p++) {
            double d = g->values[p];
            int iv = (int)d % 127;
            if (iv == 0)
                iv = 1;
            v[p] = (int8_t)iv;
        }

        raw->values = v;
    }

    return rvsp_csr_create(
        out, rows, cols, nnz,
        raw->row_ptr, raw->col_idx, raw->values, dtype);
}

/* Convert parsed Matrix Market CSR values to the selected kernel dtype. */
static int build_from_mtx(
    struct CSR *parsed,
    rvsp_dtype_t dtype,
    rvsp_csr_matrix_t *out,
    raw_csr_t *raw) {

    int32_t rows = (int32_t)vector_size(parsed->row_ptr) - 1;
    int32_t nnz  = (int32_t)vector_size(parsed->col_ind);

    const int32_t *cidx =
        (const int32_t *)parsed->col_ind->data;

    int32_t cols = 0;

    for (int32_t p = 0; p < nnz; p++)
        if (cidx[p] + 1 > cols)
            cols = cidx[p] + 1;

    raw->row_ptr = malloc((size_t)(rows + 1) * sizeof(int32_t));
    raw->col_idx = malloc((size_t)nnz * sizeof(int32_t));

    if (!raw->row_ptr || !raw->col_idx)
        return -1;

    memcpy(raw->row_ptr, parsed->row_ptr->data,
           (size_t)(rows + 1) * sizeof(int32_t));

    memcpy(raw->col_idx, parsed->col_ind->data,
           (size_t)nnz * sizeof(int32_t));

    const float *fv = (const float *)parsed->val->data;

    if (dtype == RVSP_DTYPE_FP32) {
        float *v = malloc((size_t)nnz * sizeof(float));
        if (!v)
            return -1;

        memcpy(v, fv, (size_t)nnz * sizeof(float));
        raw->values = v;
    } else if (dtype == RVSP_DTYPE_FP64) {
        double *v = malloc((size_t)nnz * sizeof(double));
        if (!v)
            return -1;

        for (int32_t p = 0; p < nnz; p++)
            v[p] = (double)fv[p];

        raw->values = v;
    } else {
        int8_t *v = malloc((size_t)nnz * sizeof(int8_t));
        if (!v)
            return -1;

        for (int32_t p = 0; p < nnz; p++) {
            int iv = (int)fv[p] % 127;
            if (iv == 0)
                iv = 1;
            v[p] = (int8_t)iv;
        }

        raw->values = v;
    }

    return rvsp_csr_create(
        out, rows, cols, nnz,
        raw->row_ptr, raw->col_idx, raw->values, dtype);
}

static void free_raw(raw_csr_t *raw) {
    free(raw->row_ptr);
    free(raw->col_idx);
    free(raw->values);

    raw->row_ptr = NULL;
    raw->col_idx = NULL;
    raw->values = NULL;
}

#define CSV_HEADER \
    "label,kernel,arm,build,march,cflags,cc_version,dtype," \
    "rows,cols,nnz_a,nnz_b,nnz_c,flops,op_mean,op_max,op_var," \
    "run,time_s,gops,correct,cycles,instructions"

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s --kernel NAME "
        "(--gen R C DENSITY SEED | --mtx A.mtx B.mtx | --mtx-sq M.mtx)\n"
        "          [--runs N] [--warmup W] [--label TAG] [--header]\n"
        "          [--arm ARM] [--build TAG] [--march FLAGS] [--cflags FLAGS]\n"
        "          [--cc-version VER]\n"
        "arms:    baseline autovec intrinsic adaptive\n"
        "kernels:\n",
        prog);

    for (int i = 0; i < N_KERNELS; i++)
        fprintf(stderr, "  %s\n", KERNELS[i].name);

#if !defined(__riscv_vector)
    fprintf(stderr,
        "  (vector kernels omitted: this binary was built without the V extension)\n");
#endif
}

/* Reject commas in metadata fields to preserve the fixed CSV schema. */
static int reject_comma(const char *val, const char *what) {
    if (val && strchr(val, ',')) {
        fprintf(stderr,
                "%s must not contain a comma (got \"%s\")\n",
                what, val);
        return 1;
    }

    return 0;
}

int main(int argc, char **argv) {
    const char *kernel_name = NULL;
    const char *label = "-";

    const char *arm = "-";
    const char *build = "-";
    const char *march = "-";
    const char *cc_version = "-";
    const char *cflags = "-";

    int runs = 10;
    int warmup = 3;
    int header = 0;

    enum {
        SRC_NONE,
        SRC_GEN,
        SRC_MTX,
        SRC_MTXSQ
    } src = SRC_NONE;

    int gen_r = 0;
    int gen_c = 0;
    int gen_seed = 0;
    double gen_density = 0.0;

    const char *mtx_a = NULL;
    const char *mtx_b = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--kernel") && i + 1 < argc)
            kernel_name = argv[++i];
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc)
            runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--label") && i + 1 < argc)
            label = argv[++i];
        else if (!strcmp(argv[i], "--arm") && i + 1 < argc)
            arm = argv[++i];
        else if (!strcmp(argv[i], "--build") && i + 1 < argc)
            build = argv[++i];
        else if (!strcmp(argv[i], "--march") && i + 1 < argc)
            march = argv[++i];
        else if (!strcmp(argv[i], "--cflags") && i + 1 < argc)
            cflags = argv[++i];
        else if (!strcmp(argv[i], "--cc-version") && i + 1 < argc)
            cc_version = argv[++i];
        else if (!strcmp(argv[i], "--header"))
            header = 1;
        else if (!strcmp(argv[i], "--gen") && i + 4 < argc) {
            src = SRC_GEN;
            gen_r = atoi(argv[++i]);
            gen_c = atoi(argv[++i]);
            gen_density = atof(argv[++i]);
            gen_seed = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--mtx") && i + 2 < argc) {
            src = SRC_MTX;
            mtx_a = argv[++i];
            mtx_b = argv[++i];
        }
        else if (!strcmp(argv[i], "--mtx-sq") && i + 1 < argc) {
            src = SRC_MTXSQ;
            mtx_a = argv[++i];
        }
        else {
            fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (header)
        printf("%s\n", CSV_HEADER);

    if (reject_comma(label, "--label") ||
        reject_comma(arm, "--arm") ||
        reject_comma(build, "--build") ||
        reject_comma(march, "--march") ||
        reject_comma(cflags, "--cflags") ||
        reject_comma(cc_version, "--cc-version"))
        return 2;

    if (!kernel_name || src == SRC_NONE) {
        usage(argv[0]);
        return 2;
    }

    const kernel_entry_t *K = find_kernel(kernel_name);

    if (!K) {
        fprintf(stderr, "no such kernel: %s\n", kernel_name);
        return 2;
    }

    /* Build input matrices. */
    rvsp_csr_matrix_t A = {0};
    rvsp_csr_matrix_t B = {0};

    raw_csr_t rawA = {0};
    raw_csr_t rawB = {0};

    csr_matrix_t gA = {0};
    csr_matrix_t gB = {0};

    struct CSR pA = {0};
    struct CSR pB = {0};

    if (src == SRC_GEN) {
        genmat_params_t p =
            genmat_default_params(gen_r, gen_c);

        p.density = gen_density;
        p.random_seed = gen_seed;
        p.cv = 0.5;
        p.min = 1;

        gA = genmat_generate_csr(p);

        genmat_params_t p2 =
            genmat_default_params(gen_c, gen_r);

        p2.density = gen_density;
        p2.random_seed = gen_seed + 1;
        p2.cv = 0.5;
        p2.min = 1;

        gB = genmat_generate_csr(p2);

        if (gA.nrows < 0 || gB.nrows < 0) {
            fprintf(stderr, "genmat failed\n");
            return 1;
        }

        if (build_from_genmat(&gA, K->dtype, &A, &rawA) != RVSP_SUCCESS ||
            build_from_genmat(&gB, K->dtype, &B, &rawB) != RVSP_SUCCESS) {
            fprintf(stderr, "build_from_genmat failed\n");
            return 1;
        }
    } else if (src == SRC_MTX) {
        pA = assemble_csr_matrix(mtx_a);
        pB = assemble_csr_matrix(mtx_b);

        if (!pA.row_ptr || !pB.row_ptr) {
            fprintf(stderr, "mtx load failed\n");
            return 1;
        }

        if (build_from_mtx(&pA, K->dtype, &A, &rawA) != RVSP_SUCCESS ||
            build_from_mtx(&pB, K->dtype, &B, &rawB) != RVSP_SUCCESS) {
            fprintf(stderr, "build_from_mtx failed\n");
            return 1;
        }
    } else {
        pA = assemble_csr_matrix(mtx_a);

        if (!pA.row_ptr) {
            fprintf(stderr, "mtx load failed\n");
            return 1;
        }

        if (build_from_mtx(&pA, K->dtype, &A, &rawA) != RVSP_SUCCESS) {
            fprintf(stderr, "build_from_mtx failed\n");
            return 1;
        }

        B = A;
    }

    if (rvsp_csr_validate(&A) != RVSP_SUCCESS ||
        rvsp_csr_validate(&B) != RVSP_SUCCESS) {
        fprintf(stderr, "csr validate failed (A or B malformed)\n");
        return 1;
    }

    /* Compute workload statistics outside the timed region. */
    int64_t *op_counts =
        (int64_t *)calloc((size_t)A.rows, sizeof(int64_t));

    if (!op_counts) {
        fprintf(stderr, "op_counts allocation failed\n");
        return 1;
    }

    double flops =
        2.0 * intermediate_products(&A, &B, op_counts);

    double op_mean;
    double op_max;
    double op_var;

    op_stats(op_counts, A.rows,
             &op_mean, &op_max, &op_var);

    /*
     * Allocate kernel workspace once and reuse it across all runs.
     * Keeping allocation outside the timed region isolates kernel execution.
     */
    void *ws = NULL;
    size_t ws_bytes = 0;

    {
        rvsp_status_t bs =
            rvsp_spgemm_buffer_size(B.cols, &ws_bytes);

        if (bs != RVSP_SUCCESS) {
            fprintf(stderr,
                    "rvsp_spgemm_buffer_size failed (%d) for b_cols=%d\n",
                    (int)bs, B.cols);
            return 1;
        }

        ws = malloc(ws_bytes);

        if (!ws) {
            fprintf(stderr,
                    "workspace allocation failed (%zu bytes)\n",
                    ws_bytes);
            return 1;
        }
    }

    /* Build the reference result outside the timed region. */
    int64_t *kernel_ops = NULL;

    {
        kernel_ops =
            (int64_t *)calloc((size_t)A.rows, sizeof(int64_t));

        if (!kernel_ops) {
            fprintf(stderr, "kernel op_counts allocation failed\n");
            return 1;
        }
    }

    rvsp_csr_matrix_t Cref = {0};

    int have_ref =
        (K->ref(&A, &B, &Cref, ws, ws_bytes, kernel_ops)
         == RVSP_SUCCESS);

    if (kernel_ops) {
        int32_t mismatch = -1;

        for (int32_t i = 0; i < A.rows; i++) {
            if (kernel_ops[i] != op_counts[i]) {
                mismatch = i;
                break;
            }
        }

        if (mismatch >= 0) {
            fprintf(stderr,
                    "warning: kernel op_counts disagrees with the harness "
                    "at row %d (%lld vs %lld)\n",
                    mismatch,
                    (long long)kernel_ops[mismatch],
                    (long long)op_counts[mismatch]);
        }

        free(kernel_ops);
    }

    /* Warm up the kernel before collecting measurements. */
    for (int w = 0; w < warmup; w++) {
        rvsp_csr_matrix_t Cw = {0};

        if (K->fn(&A, &B, &Cw, ws, ws_bytes, NULL)
            == RVSP_SUCCESS)
            rvsp_csr_destroy(&Cw);
    }

    /* Timed runs. */
    perf_init();

    for (int r = 0; r < runs; r++) {
        rvsp_csr_matrix_t C = {0};

        long long cycles = -1;
        long long insns = -1;

        /*
         * Do not request per-row op counts during timing; the workload
         * statistics were already computed outside the measured region.
         */
        perf_start();

        double t0 = now_seconds();

        rvsp_status_t st =
            K->fn(&A, &B, &C, ws, ws_bytes, NULL);

        double t1 = now_seconds();

        perf_stop(&cycles, &insns);

        double dt = t1 - t0;
        int correct = 0;
        int32_t nnz_c = 0;

        if (st == RVSP_SUCCESS) {
            nnz_c = C.nnz;
            correct = have_ref
                ? csr_equal(&C, &Cref, K->dtype)
                : -1;
        }

        double gops =
            (dt > 0.0) ? (flops / dt) / 1e9 : 0.0;

        printf(
            "%s,%s,%s,%s,%s,%s,%s,%s,"
            "%d,%d,%d,%d,%d,%.0f,%.4f,%.0f,%.4f,"
            "%d,%.9f,%.4f,%d,",
            label,
            K->name,
            arm,
            build,
            march,
            cflags,
            cc_version,
            K->dtype_str,
            A.rows,
            A.cols,
            A.nnz,
            B.nnz,
            nnz_c,
            flops,
            op_mean,
            op_max,
            op_var,
            r,
            dt,
            gops,
            correct);

        if (cycles >= 0)
            printf("%lld,", cycles);
        else
            printf(",");

        if (insns >= 0)
            printf("%lld\n", insns);
        else
            printf("\n");

        fflush(stdout);

        /* Validation occurs after timing. */
        if (st == RVSP_SUCCESS)
            rvsp_csr_destroy(&C);
    }

    perf_close();

    /* Release benchmark resources. */
    free(ws);
    free(op_counts);

    if (have_ref)
        rvsp_csr_destroy(&Cref);

    rvsp_csr_destroy(&A);

    if (src != SRC_MTXSQ)
        rvsp_csr_destroy(&B);

    free_raw(&rawA);

    if (src != SRC_MTXSQ)
        free_raw(&rawB);

    if (src == SRC_GEN) {
        genmat_free_csr(&gA);
        genmat_free_csr(&gB);
    }

    if (src == SRC_MTX || src == SRC_MTXSQ) {
        vector_free(pA.row_ptr);
        vector_free(pA.col_ind);
        vector_free(pA.val);

        if (src == SRC_MTX) {
            vector_free(pB.row_ptr);
            vector_free(pB.col_ind);
            vector_free(pB.val);
        }
    }

    return 0;
}