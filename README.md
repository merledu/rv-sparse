# RV-Sparse

Open-source RISC-V Vector (RVV) accelerated sparse linear algebra library.

## Overview

RV-Sparse implements high-performance sparse matrix operations targeting the
RISC-V Vector extension (RVV). The current focus is Sparse Matrix-Matrix
Multiplication (SpGEMM) using the Row-Wise Inner Product algorithm.

Matrices are stored in Compressed Sparse Row (CSR) format:
- values: non-zero floating point values
- col_indices: column index of each non-zero
- row_ptr: index into values where each row begins

## Building

Requirements: GCC, CMake 3.14 or higher

    mkdir build && cd build
    cmake ..
    make

## Usage

Step 1 - convert a Matrix Market (.mtx) file to CSR text files:

    python3 src/matmul/cooTOcsr.py

Outputs to MatData/: CSR_values.txt, CSR_colIdx.txt, CSR_rowPtr.txt,
CSR_redColIdx.txt, GeneralInfo.txt.

Step 2 - run SpGEMM (A x A):

    cd build
    ./SpGEMM

## Project Structure

    rv-sparse/
    ├── src/
    │   └── matmul/
    │       ├── AxBRowIP.c      # Row-wise inner product SpGEMM (C)
    │       └── cooTOcsr.py     # MTX to CSR converter (Python)
    ├── matrices/               # Input .mtx matrix files
    ├── MatData/                # Generated CSR data files
    └── CMakeLists.txt

## Contributing

This project is developed as part of the LFX Mentorship Program.
Contributions and bug reports are welcome.
