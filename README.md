# rv-sparse

Sparse linear algebra for RISC-V. RV-Sparse computes CSR sparse matrix-matrix
multiplication (SpGEMM), `C = A × B`, in FP32. It includes a scalar kernel and
three RISC-V Vector (RVV) kernels.

## Build

```bash
make clean
make
```

To cross-compile for RISC-V:

```bash
make TARGET_ARCH=riscv
```

This selects the RISC-V toolchain and builds with
`-march=rv64gcv -mabi=lp64d`.

You can also select the target explicitly with `ARCH_FLAGS`:

```bash
make ARCH_FLAGS="-march=rv64gcv -mabi=lp64d"   # scalar and RVV kernels
make ARCH_FLAGS="-march=rv64gc -mabi=lp64d"    # scalar kernel only
```

The RVV kernels are built when the target enables the RISC-V Vector extension.

## Usage

```c
#include "rv_sparse.h"

rvsp_csr_matrix_t A, B, C;

rvsp_csr_create(
    &A,
    M, K, A_nnz,
    A_row_ptr,
    A_col_idx,
    A_values,
    RVSP_DTYPE_FP32
);

rvsp_csr_create(
    &B,
    K, N, B_nnz,
    B_row_ptr,
    B_col_idx,
    B_values,
    RVSP_DTYPE_FP32
);

rvsp_spgemm_options_t options = {
    .backend = RVSP_BACKEND_SCALAR,
    .input_dtype = RVSP_DTYPE_FP32,
    .output_dtype = RVSP_DTYPE_FP32,
};

rvsp_spgemm_csr(&A, &B, &C, &options);

rvsp_csr_destroy(&C);
```

`rvsp_spgemm_csr` allocates the output matrix C. The caller is responsible for
releasing it with `rvsp_csr_destroy`.

For repeated multiplies with the same sparsity pattern, use the descriptor API
to reuse the structure analysis. It also allows a specific accumulation
strategy to be selected.

## Documentation

* [docs/api_design.md](docs/api_design.md) for the API reference
* [examples/](examples/) for worked examples

## License

GPL-3.0-only. See [LICENSE](LICENSE).


### Project by **Talha Ahmed** ([@talha-ahmed-1](https://github.com/talha-ahmed-1)) and mentees
**[C. Ramirez]** ([@crissmath](https://github.com/crissmath)),
**[Ayokunle Amodu]** ([@ayokunle321](https://github.com/ayokunle321)),
**[Hamza Mateen]** ([@HamzaMateen](https://github.com/HamzaMateen)) under the
[Linux Foundation Mentorship Program](https://mentorship.lfx.linuxfoundation.org/),
sponsored by [**RISC-V International**](https://riscv.org/).

Hardware evaluation was conducted on RISC-V development boards provided by
**[Cloud-V](https://cloud-v.co)** and the **DC-ROMA II** laptop provided by
**[DeepComputing](https://deepcomputing.io)**.
