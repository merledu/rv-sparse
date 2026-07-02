#include "mtx_to_csr_formatter.h"
#include "vec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printArray_int(vec_t *v) {
  for (vec_size_t i = 0; i < vector_size(v); i++) {
    printf("%d ", ((int *)v->data)[i]);
  }
  printf("\n");
}

void printArray_float(vec_t *v) {
  for (vec_size_t i = 0; i < vector_size(v); i++) {
    printf("%f ", ((float *)v->data)[i]);
  }
  printf("\n");
}

// ---------------------------------------------------------------------------
// internal helpers
// ---------------------------------------------------------------------------

// Read past all leading % comment lines, leave the file positioned at the
// first non-comment line (the dimensions line).
static void skip_comments(FILE *fin, char *buf, int bufsz) {
  while (fgets(buf, bufsz, fin) != NULL) {
    if (buf[0] != '%') {
      return; // buf now holds the dimensions line
    }
  }
}

// Parse the %%MatrixMarket banner to find out if this is a "pattern" file
// (no values, only structure) and whether it is "symmetric".
// Returns 1 on success, 0 if the banner is missing or malformed.
static int parse_banner(const char *buf, int *is_pattern, int *is_symmetric) {
  char object[64], format[64], field[64], symmetry[64];
  *is_pattern = 0;
  *is_symmetric = 0;

  // banner must start with %%MatrixMarket
  if (strncmp(buf, "%%MatrixMarket", 14) != 0) {
    return 0;
  }
  if (sscanf(buf, "%%%%MatrixMarket %63s %63s %63s %63s", object, format, field,
             symmetry) < 4) {
    return 0;
  }
  if (strcmp(field, "pattern") == 0) {
    *is_pattern = 1;
  }
  if (strcmp(symmetry, "symmetric") == 0) {
    *is_symmetric = 1;
  }
  return 1;
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

struct CSR assemble_csr_matrix(const char *filePath) {
  struct CSR matrix;
  matrix.val = NULL;
  matrix.col_ind = NULL;
  matrix.row_ptr = NULL;

  FILE *fin = fopen(filePath, "r");
  if (fin == NULL) {
    perror("Error opening file");
    return matrix;
  }

  char buffer[2048];

  // -- step 1: parse the %%MatrixMarket banner from the very first line --
  int is_pattern = 0, is_symmetric = 0;
  if (fgets(buffer, sizeof(buffer), fin) != NULL) {
    parse_banner(buffer, &is_pattern, &is_symmetric);
  }

  // -- step 2: skip remaining comment lines, land on the dimensions line --
  skip_comments(fin, buffer, sizeof(buffer));

  int M, N, L;
  if (sscanf(buffer, "%d %d %d", &M, &N, &L) != 3) {
    fprintf(stderr, "assemble_csr_matrix: failed to read dimensions\n");
    fclose(fin);
    return matrix;
  }

  // -- step 3: read all entries into flat COO arrays --
  // We allocate L entries up front for the stored triangle.  If the matrix
  // is symmetric we may need up to 2*L after expansion, so we'll realloc
  // then.  Using plain malloc here because we need random-access by index
  // for the prefix-sum pass, which vector_add doesn't give us cheaply.
  int *coo_row = (int *)malloc(L * sizeof(int));
  int *coo_col = (int *)malloc(L * sizeof(int));
  float *coo_val = (float *)malloc(L * sizeof(float));

  if (!coo_row || !coo_col || !coo_val) {
    fprintf(stderr, "assemble_csr_matrix: out of memory reading COO\n");
    free(coo_row);
    free(coo_col);
    free(coo_val);
    fclose(fin);
    return matrix;
  }

  int nnz = 0; // actual entries read (may be < L if file is short)
  for (int l = 0; l < L; l++) {
    int row, col;
    float val = 1.0f; // default for pattern matrices

    if (is_pattern) {
      if (fscanf(fin, "%d %d", &row, &col) != 2) {
        break;
      }
    } else {
      if (fscanf(fin, "%d %d %f", &row, &col, &val) != 3) {
        break;
      }
    }

    row -= 1; // Matrix Market is 1-indexed
    col -= 1;

    coo_row[nnz] = row;
    coo_col[nnz] = col;
    coo_val[nnz] = val;
    nnz++;
  }
  fclose(fin);

  // -- step 4: expand symmetric storage --
  // Symmetric MM files store only the lower (or upper) triangle.  Expand to
  // the full pattern by appending the mirror of every off-diagonal entry.
  if (is_symmetric) {
    int stored = nnz;
    // worst case: every entry is off-diagonal, so we need 2 * stored
    coo_row = (int *)realloc(coo_row, 2 * stored * sizeof(int));
    coo_col = (int *)realloc(coo_col, 2 * stored * sizeof(int));
    coo_val = (float *)realloc(coo_val, 2 * stored * sizeof(float));

    if (!coo_row || !coo_col || !coo_val) {
      fprintf(stderr,
              "assemble_csr_matrix: out of memory expanding symmetric\n");
      free(coo_row);
      free(coo_col);
      free(coo_val);
      return matrix;
    }

    for (int k = 0; k < stored; k++) {
      if (coo_row[k] != coo_col[k]) { // skip diagonal, it doesn't mirror
        coo_row[nnz] = coo_col[k];
        coo_col[nnz] = coo_row[k];
        coo_val[nnz] = coo_val[k];
        nnz++;
      }
    }
  }

  // -- step 5: count entries per row --
  int *row_count = (int *)calloc(M, sizeof(int));
  if (!row_count) {
    fprintf(stderr, "assemble_csr_matrix: out of memory building row_count\n");
    free(coo_row);
    free(coo_col);
    free(coo_val);
    return matrix;
  }
  for (int k = 0; k < nnz; k++) {
    row_count[coo_row[k]]++;
  }

  // -- step 6: prefix sum over row_count to build row_ptr --
  matrix.row_ptr = vector_create_int();

  vector_add_int(matrix.row_ptr, 0);

  for (int i = 0; i < M; i++) {
    int prev = ((int *)matrix.row_ptr->data)[i];
    vector_add_int(matrix.row_ptr, prev + row_count[i]);
  }
  free(row_count);

  // -- step 7: scatter COO entries into col_ind / val using a cursor array --
  // cursor[i] starts at row_ptr[i] and advances as we fill row i.
  int *cursor = (int *)malloc(M * sizeof(int));
  if (!cursor) {
    fprintf(stderr, "assemble_csr_matrix: out of memory building cursor\n");
    free(coo_row);
    free(coo_col);
    free(coo_val);
    return matrix;
  }

  for (int i = 0; i < M; i++) {
    cursor[i] = ((int *)matrix.row_ptr->data)[i];
  }

  // pre-populate col_ind and val with placeholders so we can write by index
  matrix.col_ind = vector_create_int();
  matrix.val = vector_create_float();

  for (int k = 0; k < nnz; k++) {
    vector_add_int(matrix.col_ind, 0);
    vector_add_float(matrix.val, 0.0f);
  }

  for (int k = 0; k < nnz; k++) {
    int r = coo_row[k];
    int dest = cursor[r]++;
    ((int *)matrix.col_ind->data)[dest] = coo_col[k];
    ((float *)matrix.val->data)[dest] = coo_val[k];
  }

  free(cursor);
  free(coo_row);
  free(coo_col);
  free(coo_val);

  return matrix;
}
