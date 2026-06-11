# rv-sparse Documentation

This directory contains the initial design documentation for the `rv-sparse` project.

The current work focuses on:

- Public API design
- Directory structure proposal
- CSR SpGEMM initial interface
- Backend selection model
- Project timeline

The proposed library structure is inspired by OpenBLAS/MKL-style organization, separating:

- public API headers
- core library logic
- sparse matrix formats
- optimized kernels
- benchmarks
- examples
- tests
- documentation
