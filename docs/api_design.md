# rv-sparse API Design

## Goal

The goal of the initial API is to expose a clean C interface for sparse matrix operations.

The first target is CSR SpGEMM with INT8 inputs and INT32 accumulation/output.

## Initial Operation

```text
C = A * B
```


