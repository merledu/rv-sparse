#include "../../include/rv_sparse.h"
#include <stdlib.h>
#include <time.h>

CSRMatrix_f32 *generate_random_csr_f32(int rows, int cols,
                                       float density, unsigned int seed)
{
    srand(seed);
    // Count nnz first
    int nnz = 0;
    for (int i = 0; i < rows * cols; i++)
    {
        float roll = (float)rand() / RAND_MAX;
        if (roll < density)
        {
            nnz++;
            (void)rand(); // keep RNG sequence aligned with fill pass
        }
    }
    if (nnz == 0)
        nnz = 1; // ensure at least one

    CSRMatrix_f32 *A = csr_alloc_f32(rows, cols, nnz);
    srand(seed); // reset to get same pattern

    int idx = 0;
    A->row_ptr[0] = 0;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            float roll = (float)rand() / RAND_MAX;
            if (roll < density)
            {
                A->values[idx] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                A->col_idx[idx] = c;
                idx++;
            }
        }
        A->row_ptr[r + 1] = idx;
    }
    A->nnz = idx;
    return A;
}
