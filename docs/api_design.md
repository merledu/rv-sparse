# rv-sparse API Design

## Goal

The goal of the initial API is to expose a clean C interface for sparse matrix operations.

The first target is CSR SpGEMM with INT8 inputs and INT32 accumulation/output.

## Initial Operation

```text
C = A * B
```

Initial data type:

INT8 X INT8 --> INT32
> Avoid early quantization complexity and gives clear accumulation.( avoid overflow)

## Main API Types

### rvsp_csr_matrix_t

Represents a sparse matrix in CSR format.

Fields:

- rows
- cols
- nnz
- row_ptr
- col_idx
- values
- dtype
- format

### rvsp_spgemm_options_t

Controls backend and data type selection.

Fields:

- backend
- input_dtype
- output_dtype
- sort_output_indices
- Backends

The API is designed to support three initial backends:

```text
RVSP_BACKEND_SCALAR
RVSP_BACKEND_GCC_AUTOVEC
RVSP_BACKEND_RVV_INTRINSICS
```

These map directly to the project evaluation table:

Scalar                     -> 1.0x
GCC Auto-Vectorized        -> Xx
Hand-written RVV Intrinsics -> Yx

## Public Functions

rvsp_csr_create()
Initializes a CSR matrix descriptor.

rvsp_csr_validate()
Validates CSR metadata and indexing structure.

rvsp_csr_destroy()
Resets a CSR matrix descriptor.

rvsp_spgemm_csr()
Computes sparse matrix-matrix multiplication using the selected backend.

## Ownership Model

The initial API does not take ownership of user-provided arrays.

The caller remains responsible for allocating and freeing:

- row_ptr
- col_idx
- values

> note:This keeps the first version simple and predictable.

## Future Extensions

### Planned data types

BF16
FP32

### Planned formats

- CSR
- COO
- SELL-C

### Planned improvements

- Runtime backend dispatch
- RVV optimized kernels
- Benchmark harness
- More complete memory ownership options
