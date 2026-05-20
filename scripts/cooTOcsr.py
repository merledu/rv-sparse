import sys
import os

def ProcessMTX(mtx_path):
    matrix = []
    rows = []
    cols = []
    values = []
    no_of_rows = 0
    no_of_cols = 0
    NNZ_values = 0
    MAC_count = 0
    row_col = {}
    rc_value = {}
    
    with open(mtx_path, 'r') as f:
        matrix = f.readlines()
    
    isSymmetric = matrix[0].split(" ")[-1] == "symmetric\n"
    isPattern = "pattern" in matrix[0].split(" ")
    
    isFirstLine = True
    for i in matrix:           
        if i[0] != "%":
            if isFirstLine:
                no_of_rows = int(i.split(" ")[0])
                no_of_cols = int(i.split(" ")[1])
                NNZ_values = int(i.split(" ")[2]) if not isSymmetric else int(i.split(" ")[2])*2
                isFirstLine = False
            else:
                row_item = int(i.split(" ")[0])
                col_item = int(i.split(" ")[1][:-1]) if isPattern else int(i.split(" ")[1])
                val_item = float(1) if isPattern else float(i.split(" ")[2][:-1])
                
                rows.append(row_item-1)
                cols.append(col_item-1)
                values.append(val_item)
                
                rc_value[row_item] = [col_item, val_item]
    
                if row_item in row_col:
                    row_col[row_item] += [col_item]
                else:
                    row_col[row_item] = [col_item]
    
                if isSymmetric:
                    if col_item in row_col:
                        row_col[col_item] += [row_item]
                    else:
                        row_col[col_item] = [row_item]
                    
                    # For SpGEMM, if symmetric, we also need to append the transposed entry.
                    if row_item != col_item:
                        rows.append(col_item-1)
                        cols.append(row_item-1)
                        values.append(val_item)

    # Recalculate NNZ in case we appended symmetric entries
    NNZ_values = len(rows)

    return rows, cols, values, no_of_rows, no_of_cols, NNZ_values

def coo_to_csr(coo_row, coo_col, coo_data, nrows):
    # Zip, sort by row first, then by col, to ensure CSR column indices are sorted within each row.
    # This solves the correctness bug where unordered COO yields unsorted CSR.
    sorted_triplets = sorted(zip(coo_row, coo_col, coo_data), key=lambda x: (x[0], x[1]))
    if sorted_triplets:
        coo_row, coo_col, coo_data = zip(*sorted_triplets)
    else:
        coo_row, coo_col, coo_data = [], [], []

    # Initialize the CSR arrays
    csr_data = []
    csr_indices = []
    csr_indptr = [0]  # Starts with 0

    # Count the non-zero elements in each row
    row_counts = [0] * (nrows+1)
    for row in coo_row:
        row_counts[row] += 1

    # Fill the csr_indptr array based on row_count
    for i in range(nrows):
        csr_indptr.append(csr_indptr[-1] + row_counts[i])

    # Fill the csr_data and csr_indices arrays
    for i in range(len(coo_data)):
        col = coo_col[i]
        data = coo_data[i]

        csr_indices.append(col)
        csr_data.append(data)

    # Fill the reduced csr_indices
    reduced_indices = []
    indx1 = 0
    indx2 = 1
    while indx2 != len(csr_indptr):
        count = 0
        for i in range(csr_indptr[indx1], csr_indptr[indx2]):
            reduced_indices.append(count)
            count += 1
        indx1 += 1
        indx2 += 1
    return csr_data, csr_indices, csr_indptr, reduced_indices
    

def Main(mat_A, mat_B):
    matrix_list = ["patents_main",
                   "p2p-Gnutella31",
                   "roadNet-CA",
                   "webbase-1M",
                   "m133-b3",
                   "cit-Patents",
                   "mario002",
                   "web-Google",
                   "scircuit",
                   "amazon0312",
                   "ca-CondMat",
                   "email-Enron",
                   "wiki-Vote",
                   "cage12",
                   "2cubes_sphere",
                   "offshore",
                   "cop20k_A",
                   "filter3D",
                   "poisson3Da",
                   "example"]

    # Map numeric index to name if string is convertible to int and within range
    try:
        idx_A = int(mat_A)
        if 0 <= idx_A < len(matrix_list):
            mat_A = matrix_list[idx_A]
    except ValueError:
        pass

    try:
        idx_B = int(mat_B)
        if 0 <= idx_B < len(matrix_list):
            mat_B = matrix_list[idx_B]
    except ValueError:
        pass

    mtx_path_A = f"matrices/{mat_A}/{mat_A}.mtx"
    mtx_path_B = f"matrices/{mat_B}/{mat_B}.mtx"

    if not os.path.exists(mtx_path_A):
        print(f"Error: Matrix A path {mtx_path_A} does not exist.")
        return
    if not os.path.exists(mtx_path_B):
        print(f"Error: Matrix B path {mtx_path_B} does not exist.")
        return

    print(f"Processing Matrix A: {mtx_path_A}")
    rows_A, cols_A, values_A, M, K_A, aNNZ = ProcessMTX(mtx_path_A)
    CSR_values_A, CSR_colIdx_A, CSR_rowPtr_A, CSR_redColIdx_A = coo_to_csr(rows_A, cols_A, values_A, M)

    print(f"Processing Matrix B: {mtx_path_B}")
    rows_B, cols_B, values_B, K_B, N, bNNZ = ProcessMTX(mtx_path_B)
    CSR_values_B, CSR_colIdx_B, CSR_rowPtr_B, CSR_redColIdx_B = coo_to_csr(rows_B, cols_B, values_B, K_B)

    if K_A != K_B:
        print(f"Warning: Dimensions mismatch for SpGEMM! A is {M}x{K_A}, B is {K_B}x{N}")

    os.makedirs('matrix_data', exist_ok=True)

    # Export Matrix A CSR
    with open('matrix_data/A_values.txt', 'w') as f:
        for item in CSR_values_A:
            f.write(f"{item}\n")
    
    with open('matrix_data/A_colIdx.txt', 'w') as f:
        for item in CSR_colIdx_A:
            f.write(f"{item}\n")
    
    with open('matrix_data/A_rowPtr.txt', 'w') as f:
        for item in CSR_rowPtr_A:
            f.write(f"{item}\n")

    # Export Matrix B CSR
    with open('matrix_data/B_values.txt', 'w') as f:
        for item in CSR_values_B:
            f.write(f"{item}\n")
    
    with open('matrix_data/B_colIdx.txt', 'w') as f:
        for item in CSR_colIdx_B:
            f.write(f"{item}\n")
    
    with open('matrix_data/B_rowPtr.txt', 'w') as f:
        for item in CSR_rowPtr_B:
            f.write(f"{item}\n")
            
    # Export GeneralInfo: M, K_A, N, aNNZ, bNNZ
    with open('matrix_data/GeneralInfo.txt', 'w') as f:
        for item in [M, K_A, N, aNNZ, bNNZ]:
            f.write(f"{item}\n")
    print(f"Successfully generated CSR matrices in matrix_data/: A ({M}x{K_A}, NNZ={aNNZ}), B ({K_B}x{N}, NNZ={bNNZ})")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        Main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        Main(sys.argv[1], sys.argv[1])
    else:
        # Default fallback to example if it exists, or email-Enron (which needs downloading).
        # We'll default to 'example_square_A' and 'example_square_B' as our standard test matrices.
        if os.path.exists("matrices/example_square_A/example_square_A.mtx"):
            Main("example_square_A", "example_square_B")
        else:
            Main(11, 11)