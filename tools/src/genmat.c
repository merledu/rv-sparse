#include "genmat.h"

/* The omp pragmas below are ignored without -fopenmp, but the header only
 * exists when the toolchain ships OpenMP. Build with OPENMP=1 to enable. */
#ifdef _OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define USHI unsigned short int
#define INTSIZE 268435456

// ---------------------------------------------------------------------------
// internal helpers (unchanged from the original CLI tool, just made static)
// ---------------------------------------------------------------------------

static void *safe_malloc(size_t size) {
  void *loc = malloc(size);
  if (loc == NULL) {
    fprintf(stderr,
            "genmat.c: safe_malloc: Memory (%lu = %lu GB) could not be "
            "allocated\n",
            (unsigned long)size, (unsigned long)(size / INTSIZE));
    exit(1);
  }
  return loc;
}

static void *safe_calloc(size_t count, size_t size) {
  void *loc = calloc(count, size);
  if (loc == NULL) {
    fprintf(stderr,
            "genmat.c: safe_calloc: Memory (%lu = %lu GB) could not be "
            "(c)allocated\n",
            (unsigned long)size, (unsigned long)(size / INTSIZE));
    exit(1);
  }
  return loc;
}

// Explicit-state PRNG, plays the same role rand_r() did (caller owns the
// state, so it's safe to call from multiple OpenMP threads with their own
// seeds) but is plain C11 with no POSIX dependency.
static unsigned int xorshift32(unsigned int *state) {
  unsigned int x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return *state = x;
}

static double norm_box_muller(double mean, double stdev, int seed_bm) {
  double u, r, theta;
  double x;
  double norm_rv;

  unsigned int mystate = seed_bm * 10;
  if (mystate == 0)
    mystate = 0x9e3779b9;

  u = 0.0;
  while (u == 0.0) {
    u = (double)xorshift32(&mystate) /
        4294967295.0; // UINT_MAX, full xorshift32 range
  }

  r = sqrt(-2.0 * log(u));

  mystate = floor(mean) * seed_bm;
  if (mystate == 0)
    mystate = 0x9e3779b9;

  theta = 0.0;
  while (theta == 0.0) {
    theta = 6.2831853 * xorshift32(&mystate) / 4294967295.0;
  }

  x = r * cos(theta);
  norm_rv = (x * stdev) + mean;

  return (norm_rv);
}

static int rand_count(double avg, double std, int min, int max, int curr_seed,
                      double vals_pos) {
  int curr_degree;

  if (vals_pos > 0) {
    curr_degree = (int)round(norm_box_muller(avg, std, curr_seed));
  } else {
    double avg_log_norm = log(avg * avg / sqrt(avg * avg + std * std));
    double std_log_norm = sqrt(log(1 + (std * std) / (avg * avg)));

    curr_degree =
        (int)round(exp(norm_box_muller(avg_log_norm, std_log_norm, curr_seed)));
  }

  if (curr_degree < min) {
    curr_degree = min;
  }
  if (curr_degree > max) {
    curr_degree = max;
  }

  return curr_degree;
}

// ---------------------------------------------------------------------------
// distribution functions: fill degrees[]/indices[][] with the nonzero
// pattern. Logic is identical to the original tool, only made static.
// ---------------------------------------------------------------------------

static void distribute_sym_case1(double avg, double std, int min, int max,
                                 int id_first, int random_seed, int seed_mult,
                                 int low_bandwidth, int *degrees,
                                 int **indices) {
  int r1 = id_first - 1;
  int r2 = id_first / 2;
  int is_r_odd = id_first % 2;

  double vals_pos = avg - 3 * std;

#pragma omp parallel
  {
    USHI *is_nz_ind = (USHI *)safe_malloc(id_first * sizeof(USHI));
#pragma omp for
    for (int j = 0; j < low_bandwidth; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed
      int jl = j + low_bandwidth;

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % jl] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));
      indices[r1 - j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < j; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      indices[j][curr_degree++] = j;
      degrees[j] = curr_degree;

      curr_degree = 0;
      for (int k = j + 1; k < j + low_bandwidth; k++) {
        if (is_nz_ind[k]) {
          indices[r1 - j][curr_degree++] = id_first - k;
        }
      }
      indices[r1 - j][curr_degree++] = r1 - j;
      degrees[r1 - j] = curr_degree;
    }

#pragma omp for
    for (int j = low_bandwidth; j < r2; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed
      int jl = j - low_bandwidth;
      int l2 = 2 * low_bandwidth;

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % l2 + jl] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));
      indices[r1 - j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = j - low_bandwidth; k < j; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      indices[j][curr_degree++] = j;
      degrees[j] = curr_degree;

      curr_degree = 0;
      for (int k = j + 1; k < j + low_bandwidth; k++) {
        if (is_nz_ind[k]) {
          indices[r1 - j][curr_degree++] = id_first - k;
        }
      }
      indices[r1 - j][curr_degree++] = r1 - j;
      degrees[r1 - j] = curr_degree;
    }
    free(is_nz_ind);
  }

  if (is_r_odd) {
    USHI *is_nz_ind = (USHI *)safe_malloc(id_first * sizeof(USHI));
    int j = r2 + 1;

    int curr_seed = random_seed * (j + seed_mult);
    int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

#pragma omp parallel for
    for (int k = 0; k < id_first; k++) {
      is_nz_ind[k] = 0;
    }

    int jl = j - low_bandwidth;
    int l2 = 2 * low_bandwidth;

    for (int k = 0; k < curr_degree - 1; k++) {
      is_nz_ind[rand() % l2 + jl] = 1;
    }

    indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));
    indices[r1 - j] = (int *)safe_calloc(curr_degree, sizeof(int));

    curr_degree = 0;
    for (int k = j - low_bandwidth; k < j; k++) {
      if (is_nz_ind[k]) {
        indices[j][curr_degree++] = k;
      }
    }
    indices[j][curr_degree++] = j;
    degrees[j] = curr_degree;
    free(is_nz_ind);
  }
}

static void distribute_sym_case2(double avg, double std, int min, int max,
                                 int id_first, int random_seed, int seed_mult,
                                 int low_bandwidth, int *degrees,
                                 int **indices) {
  int r1 = id_first - 1;
  int r2 = id_first / 2;
  int is_r_odd = id_first % 2;
  int rl = id_first - low_bandwidth;

  double vals_pos = avg - 3 * std;

#pragma omp parallel
  {
    USHI *is_nz_ind = (USHI *)safe_malloc(id_first * sizeof(USHI));
#pragma omp for
    for (int j = 0; j < rl - 1; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed
      int jl = j + low_bandwidth;

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % jl] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));
      indices[r1 - j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < j; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      indices[j][curr_degree++] = j;
      degrees[j] = curr_degree;

      curr_degree = 0;
      for (int k = j + 1; k < j + low_bandwidth; k++) {
        if (is_nz_ind[k]) {
          indices[r1 - j][curr_degree++] = id_first - k;
        }
      }
      indices[r1 - j][curr_degree++] = r1 - j;
      degrees[r1 - j] = curr_degree;
    }

#pragma omp for
    for (int j = rl - 1; j < r2; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % id_first] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));
      indices[r1 - j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < j; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      indices[j][curr_degree++] = j;
      degrees[j] = curr_degree;

      curr_degree = 0;
      for (int k = j + 1; k < id_first; k++) {
        if (is_nz_ind[k]) {
          indices[r1 - j][curr_degree++] = id_first - k;
        }
      }
      indices[r1 - j][curr_degree++] = r1 - j;
      degrees[r1 - j] = curr_degree;
    }
    free(is_nz_ind);
  }

  if (is_r_odd) {
    USHI *is_nz_ind = (USHI *)safe_malloc(id_first * sizeof(USHI));
    int j = r2 + 1;
    int curr_seed = random_seed * (j + seed_mult);
    int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

#pragma omp parallel for
    for (int k = 0; k < id_first; k++) {
      is_nz_ind[k] = 0;
    }

    for (int k = 0; k < curr_degree - 1; k++) {
      is_nz_ind[rand() % id_first] = 1;
    }

    indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));
    indices[r1 - j] = (int *)safe_calloc(curr_degree, sizeof(int));

    curr_degree = 0;
    for (int k = 0; k < j; k++) {
      if (is_nz_ind[k]) {
        indices[j][curr_degree++] = k;
      }
    }
    indices[j][curr_degree++] = j;
    degrees[j] = curr_degree;
    free(is_nz_ind);
  }
}

static void distribute_square_nonsym_case1(double avg, double std, int min,
                                           int max, int id_first,
                                           int random_seed, int seed_mult,
                                           int low_bandwidth, int up_bandwidth,
                                           int *degrees, int **indices) {
  int ru = id_first - up_bandwidth;
  double vals_pos = avg - 3 * std;

#pragma omp parallel
  {
    USHI *is_nz_ind = (USHI *)safe_malloc(id_first * sizeof(USHI));
#pragma omp for
    for (int j = 0; j < ru; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }
      is_nz_ind[j] = 1;

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed
      int ju = j + up_bandwidth;

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % ju] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < id_first; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      degrees[j] = curr_degree;
    }

#pragma omp for
    for (int j = ru; j < low_bandwidth; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }
      is_nz_ind[j] = 1;

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % id_first] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < id_first; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      degrees[j] = curr_degree;
    }

#pragma omp for
    for (int j = low_bandwidth; j < id_first; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }
      is_nz_ind[j] = 1;

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed
      int jl = j - low_bandwidth;
      int ijl = id_first - jl;

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % ijl + jl] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < id_first; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      degrees[j] = curr_degree;
    }
    free(is_nz_ind);
  }
}

static void distribute_square_nonsym_case2(double avg, double std, int min,
                                           int max, int id_first,
                                           int random_seed, int seed_mult,
                                           int low_bandwidth, int up_bandwidth,
                                           int *degrees, int **indices) {
  int ul = up_bandwidth + low_bandwidth;
  int ru = id_first - up_bandwidth;
  double vals_pos = avg - 3 * std;

#pragma omp parallel
  {
    USHI *is_nz_ind = (USHI *)safe_malloc(id_first * sizeof(USHI));
#pragma omp for
    for (int j = 0; j < low_bandwidth; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }
      is_nz_ind[j] = 1;

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed
      int ju = j + up_bandwidth;

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % ju] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < id_first; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      degrees[j] = curr_degree;
    }

#pragma omp for
    for (int j = low_bandwidth; j < ru; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }
      is_nz_ind[j] = 1;

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed
      int jl = j - low_bandwidth;

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % ul + jl] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < id_first; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      degrees[j] = curr_degree;
    }

#pragma omp for
    for (int j = ru; j < id_first; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_first; k++) {
        is_nz_ind[k] = 0;
      }
      is_nz_ind[j] = 1;

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed
      int jl = j - low_bandwidth;
      int ijl = id_first - jl;

      for (int k = 0; k < curr_degree - 1; k++) {
        is_nz_ind[xorshift32(&mystate) % ijl + jl] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < id_first; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree++] = k;
        }
      }
      degrees[j] = curr_degree;
    }
    free(is_nz_ind);
  }
}

static void distribute_rectangular(double avg, double std, int min, int max,
                                   int id_first, int id_second, int random_seed,
                                   int seed_mult, int *degrees, int **indices) {
  double vals_pos = avg - 3 * std;

#pragma omp parallel
  {
    USHI *is_nz_ind = (USHI *)safe_malloc(id_second * sizeof(USHI));
#pragma omp for
    for (int j = 0; j < id_first; j++) {
      int curr_seed = random_seed * (j + seed_mult);
      int curr_degree = rand_count(avg, std, min, max, curr_seed, vals_pos);

      for (int k = 0; k < id_second; k++) {
        is_nz_ind[k] = 0;
      }

      unsigned int mystate = curr_seed + curr_degree % 10;
      if (mystate == 0)
        mystate = 0x9e3779b9; // xorshift32 cannot recover from a zero seed

      for (int k = 0; k < curr_degree; k++) {
        is_nz_ind[xorshift32(&mystate) % id_second] = 1;
      }

      indices[j] = (int *)safe_calloc(curr_degree, sizeof(int));

      curr_degree = 0;
      for (int k = 0; k < id_second; k++) {
        if (is_nz_ind[k]) {
          indices[j][curr_degree] = k;
          curr_degree++;
        }
      }
      degrees[j] = curr_degree;
    }
    free(is_nz_ind);
  }
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

genmat_params_t genmat_default_params(int row_cnt, int col_cnt) {
  genmat_params_t p;
  p.row_cnt = row_cnt;
  p.col_cnt = col_cnt;
  p.density = 0.001;
  p.cv = 0.5;
  p.min = 1;
  p.imbalance = -1.0;
  p.is_symmetric = 0;
  p.is_column = 0;
  p.low_bandwidth = row_cnt - 1;
  p.up_bandwidth = row_cnt - 1;
  p.random_seed = 1;
  return p;
}

// Converts (coo_row, coo_col, coo_val) triples of length nnz into a CSR
// matrix: counting sort by row, then ascending sort of columns within each
// row. Does not free the coo_* arrays, caller's responsibility.
static void sort_row(int *col_idx, double *values, int start, int end);

static csr_matrix_t coo_to_csr(int nrows, int ncols, long long nnz,
                               const int *coo_row, const int *coo_col,
                               const double *coo_val) {
  int *row_ptr = (int *)safe_calloc(nrows + 1, sizeof(int));
  for (long long k = 0; k < nnz; k++) {
    row_ptr[coo_row[k] + 1]++;
  }
  for (int i = 0; i < nrows; i++) {
    row_ptr[i + 1] += row_ptr[i];
  }

  int *col_idx = (int *)safe_malloc(nnz * sizeof(int));
  double *values = (double *)safe_malloc(nnz * sizeof(double));
  int *cursor = (int *)safe_malloc(nrows * sizeof(int));
  memcpy(cursor, row_ptr, nrows * sizeof(int));

  for (long long k = 0; k < nnz; k++) {
    int r = coo_row[k];
    int dest = cursor[r]++;
    col_idx[dest] = coo_col[k];
    values[dest] = coo_val[k];
  }
  free(cursor);

  for (int i = 0; i < nrows; i++) {
    sort_row(col_idx, values, row_ptr[i], row_ptr[i + 1]);
  }

  // Some of the distribute_* functions can emit the same (row,col) twice
  // (most notably the symmetric mirror-row diagonal, see symmetrize_pattern
  // comment). Merge duplicates within a row by summing, in place, the same
  // convention most COO->CSR converters use.
  int *deduped_row_ptr = (int *)safe_calloc(nrows + 1, sizeof(int));
  long long write = 0;
  for (int i = 0; i < nrows; i++) {
    int start = row_ptr[i], end = row_ptr[i + 1];
    deduped_row_ptr[i] = (int)write;
    int k = start;
    while (k < end) {
      int col = col_idx[k];
      double sum = values[k];
      int k2 = k + 1;
      while (k2 < end && col_idx[k2] == col) {
        sum += values[k2];
        k2++;
      }
      col_idx[write] = col;
      values[write] = sum;
      write++;
      k = k2;
    }
  }
  deduped_row_ptr[nrows] = (int)write;
  free(row_ptr);
  row_ptr = deduped_row_ptr;
  nnz = write;

  csr_matrix_t m;
  m.nrows = nrows;
  m.ncols = ncols;
  m.nnz = nnz;
  m.row_ptr = row_ptr;
  m.col_idx = col_idx;
  m.values = values;
  return m;
}

// binary search row r (sorted ascending) for column c; returns index into
// col_idx/values if found, -1 otherwise.
static int find_in_row(const int *row_ptr, const int *col_idx, int r, int c) {
  int lo = row_ptr[r], hi = row_ptr[r + 1] - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (col_idx[mid] == c) {
      return mid;
    } else if (col_idx[mid] < c) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return -1;
}

// Ensures the pattern is structurally symmetric: for every stored (i,j),
// guarantees (j,i) is also stored, inserting it (with the same value, the
// later symmetrize_values pass reconciles cases where both sides already
// existed independently) if it's missing. Frees the input matrix and
// returns a new one; if the pattern was already symmetric, returns the
// input unchanged.
static csr_matrix_t symmetrize_pattern(csr_matrix_t m) {
  long long missing = 0;
  for (int i = 0; i < m.nrows; i++) {
    for (int k = m.row_ptr[i]; k < m.row_ptr[i + 1]; k++) {
      int j = m.col_idx[k];
      if (j == i) {
        continue;
      }
      if (find_in_row(m.row_ptr, m.col_idx, j, i) < 0) {
        missing++;
      }
    }
  }

  if (missing == 0) {
    return m;
  }

  long long new_nnz = m.nnz + missing;
  int *coo_row = (int *)safe_malloc(new_nnz * sizeof(int));
  int *coo_col = (int *)safe_malloc(new_nnz * sizeof(int));
  double *coo_val = (double *)safe_malloc(new_nnz * sizeof(double));

  long long pos = 0;
  for (int i = 0; i < m.nrows; i++) {
    for (int k = m.row_ptr[i]; k < m.row_ptr[i + 1]; k++) {
      coo_row[pos] = i;
      coo_col[pos] = m.col_idx[k];
      coo_val[pos] = m.values[k];
      pos++;
    }
  }
  for (int i = 0; i < m.nrows; i++) {
    for (int k = m.row_ptr[i]; k < m.row_ptr[i + 1]; k++) {
      int j = m.col_idx[k];
      if (j == i) {
        continue;
      }
      if (find_in_row(m.row_ptr, m.col_idx, j, i) < 0) {
        coo_row[pos] = j;
        coo_col[pos] = i;
        coo_val[pos] = m.values[k]; // mirror the source value
        pos++;
      }
    }
  }

  int nrows = m.nrows, ncols = m.ncols;
  genmat_free_csr(&m);

  csr_matrix_t result =
      coo_to_csr(nrows, ncols, new_nnz, coo_row, coo_col, coo_val);
  free(coo_row);
  free(coo_col);
  free(coo_val);
  return result;
}

static csr_matrix_t empty_csr_error(void) {
  csr_matrix_t m;
  m.nrows = -1;
  m.ncols = -1;
  m.nnz = 0;
  m.row_ptr = NULL;
  m.col_idx = NULL;
  m.values = NULL;
  return m;
}

// Insertion sort of (col_idx, values) for one row, ascending by col_idx.
// nnz per row is small (it's `avg` nonzeros per row), so this is cheap and
// avoids pulling in qsort with a comparator + index-pairing struct.
static void sort_row(int *col_idx, double *values, int start, int end) {
  for (int i = start + 1; i < end; i++) {
    int key_col = col_idx[i];
    double key_val = values[i];
    int j = i - 1;
    while (j >= start && col_idx[j] > key_col) {
      col_idx[j + 1] = col_idx[j];
      values[j + 1] = values[j];
      j--;
    }
    col_idx[j + 1] = key_col;
    values[j + 1] = key_val;
  }
}

// For symmetric matrices the pattern already has both (i,j) and (j,i)
// stored, but each was assigned an independent random value. This walks
// rows in ascending order and, for every entry below the diagonal, copies
// the value already generated for its mirror above the diagonal (which is
// guaranteed to exist and to have been visited already, since j < i).
static void symmetrize_values(int nrows, const int *row_ptr, int *col_idx,
                              double *values) {
  for (int i = 0; i < nrows; i++) {
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      int j = col_idx[k];
      if (j >= i) {
        continue; // diagonal or upper triangle: this is the canonical value
      }
      // binary search row j (sorted ascending) for column i
      int lo = row_ptr[j], hi = row_ptr[j + 1] - 1;
      while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (col_idx[mid] == i) {
          values[k] = values[mid];
          break;
        } else if (col_idx[mid] < i) {
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
    }
  }
}

csr_matrix_t genmat_generate_csr(genmat_params_t params) {
  int row_cnt = params.row_cnt;
  int col_cnt = params.col_cnt;
  int is_square = (row_cnt == col_cnt);
  int is_symmetric = params.is_symmetric;
  int is_column = params.is_column;
  int low_bandwidth = params.low_bandwidth;
  int up_bandwidth = params.up_bandwidth;
  int random_seed = params.random_seed;
  double density = params.density;
  double cv = params.cv;
  int min = params.min;
  double imbalance = params.imbalance;

  if (is_symmetric && !is_square) {
    fprintf(stderr, "genmat_generate_csr: symmetric requested but matrix "
                    "is not square\n");
    return empty_csr_error();
  }

  srand(random_seed);

  if (is_symmetric) {
    up_bandwidth = low_bandwidth;
  }

  int id_first = row_cnt;
  int id_second = col_cnt;
  if (is_column && !is_square) {
    id_first = col_cnt;
    id_second = row_cnt;
  }

  double avg = density * id_second;
  double std = cv * avg;

  int max = id_second - 1;
  if (imbalance != -1.0) {
    max = (int)round(imbalance * avg + avg);
  }
  if (max > id_second - 1) {
    max = id_second - 1;
  }

  int **indices = (int **)safe_malloc(id_first * sizeof(int *));
  int *degrees = (int *)safe_malloc(id_first * sizeof(int));

  // -- phase 1: decide nonzero pattern (degrees[] / indices[][]) --
  if (is_square) {
    if (is_symmetric) {
      int r2 = row_cnt / 2;
      if (low_bandwidth < r2) {
        distribute_sym_case1(avg, std, min, max, id_first, random_seed, 2,
                             low_bandwidth, degrees, indices);
      } else {
        distribute_sym_case2(avg, std, min, max, id_first, random_seed, 3,
                             low_bandwidth, degrees, indices);
      }
    } else {
      int ru = row_cnt - up_bandwidth;
      if (low_bandwidth > ru) {
        distribute_square_nonsym_case1(avg, std, min, max, id_first,
                                       random_seed, 4, low_bandwidth,
                                       up_bandwidth, degrees, indices);
      } else {
        distribute_square_nonsym_case2(avg, std, min, max, id_first,
                                       random_seed, 5, low_bandwidth,
                                       up_bandwidth, degrees, indices);
      }
    }
  } else {
    distribute_rectangular(avg, std, min, max, id_first, id_second, random_seed,
                           6, degrees, indices);
  }

  // -- phase 2: total nnz, same as the original tool recomputed it --
  long long total_nnz = 0;
  for (int j = 0; j < id_first; j++) {
    total_nnz += degrees[j];
  }

  // -- phase 3: emit (row, col, value) triples, same iteration order and
  //    same rand() value formula the old Matrix Market writer used --
  int *coo_row = (int *)safe_malloc(total_nnz * sizeof(int));
  int *coo_col = (int *)safe_malloc(total_nnz * sizeof(int));
  double *coo_val = (double *)safe_malloc(total_nnz * sizeof(double));

  long long pos = 0;
  if (!is_column) {
    for (int i = 0; i < row_cnt; i++) {
      for (int j = 0; j < degrees[i]; j++) {
        coo_row[pos] = i;
        coo_col[pos] = indices[i][j];
        coo_val[pos] = (rand() % 9 + 1.0) / 10;
        pos++;
      }
    }
  } else {
    for (int i = 0; i < col_cnt; i++) {
      for (int j = 0; j < degrees[i]; j++) {
        coo_row[pos] = indices[i][j];
        coo_col[pos] = i;
        coo_val[pos] = (rand() % 9 + 1.0) / 10;
        pos++;
      }
    }
  }

  for (int i = 0; i < id_first; i++) {
    free(indices[i]);
  }
  free(indices);
  free(degrees);

  // -- phase 4: COO -> CSR --
  csr_matrix_t m =
      coo_to_csr(row_cnt, col_cnt, total_nnz, coo_row, coo_col, coo_val);
  free(coo_row);
  free(coo_col);
  free(coo_val);

  if (is_symmetric) {
    m = symmetrize_pattern(m); // guarantee (i,j) stored iff (j,i) stored
    symmetrize_values(m.nrows, m.row_ptr, m.col_idx, m.values); // and equal
  }

  return m;
}

void genmat_free_csr(csr_matrix_t *m) {
  free(m->row_ptr);
  free(m->col_idx);
  free(m->values);
  m->row_ptr = NULL;
  m->col_idx = NULL;
  m->values = NULL;
  m->nrows = 0;
  m->ncols = 0;
  m->nnz = 0;
}

// export/save to file
int genmat_save_csr_to_mtx(const csr_matrix_t *csr, const char *filename) {
  FILE *f = fopen(filename, "w");
  if (!f)
    return -1;

  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%d %d %lld\n", csr->nrows, csr->ncols, csr->nnz);

  for (int i = 0; i < csr->nrows; i++) {
    for (int ptr = csr->row_ptr[i]; ptr < csr->row_ptr[i + 1]; ptr++) {
      fprintf(f, "%d %d %.16e\n", i + 1, csr->col_idx[ptr] + 1,
              csr->values[ptr]);
    }
  }

  fclose(f);
  return 0;
}
