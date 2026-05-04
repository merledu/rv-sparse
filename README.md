# RV-Sparse

Open-source RISC-V Vector accelerated sparse linear algebra library.

## Build and test (native)

Requires a C11 toolchain (`cc`). From the repository root:

```bash
make check    # CSR SpMM regression tests (dense reference comparison)
make all      # builds unit tests and demo driver ./axb_rowip
```

`matrix_data/GeneralInfo.txt`: either **four** integers (`M`, `K`, `aNNZ`, `rN`, legacy square demo with `N = M`) or **five** (`M`, `K`, `N`, `aNNZ`, `rN`) for rectangular outputs.

Run the demo from the repo root (loads `matrix_data/*.txt`):

```bash
./axb_rowip
```
