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
                
                # rc_value[f"{row_item}_{col_item}"] = val_item
                rc_value[row_item] = [col_item, val_item]
    
                # if row_item not in CSR_rowPtr:
                #     CSR_rowPtr.append(
                
                if row_item in row_col:
                    row_col[row_item] += [col_item]
                else:
                    row_col[row_item] = [col_item]
    
                if isSymmetric:
                    if col_item in row_col:
                        row_col[col_item] += [row_item]
                    else:
                        row_col[col_item] = [row_item]

    return rows,cols,values,no_of_rows,no_of_cols,NNZ_values

def coo_to_csr(coo_row, coo_col, coo_data, nrows):
    # Initialize the CSR arrays
    csr_data = []
    csr_indices = []
    csr_indptr = [0]  # Starts with 0

    # Count the non-zero elements in each row
    row_counts = [0] * (nrows+1)
    for row in coo_row:
        # print(row)
        row_counts[row] += 1

    # Fill the csr_indptr array based on row_count
    for i in range(nrows):
        csr_indptr.append(csr_indptr[-1] + row_counts[i])
    # print(csr_indptr)

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
    

def Main(matID):
    matrix = ["patents_main",
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

    mtx_path = f"matrices/{matrix[matID]}/"+matrix[matID]+".mtx"

    rows,cols,values,no_of_rows,no_of_cols,NNZ_values = ProcessMTX(mtx_path)
    CSR_values,CSR_colIdx,CSR_rowPtr, CSR_redColIdx = coo_to_csr(rows,cols,values,no_of_rows)

    with open('matrix_data/CSR_values.txt', 'w') as f:
        for item in CSR_values:
            f.write(f"{item}\n")
        # f.write(f"{CSR_values}")
    
    with open('matrix_data/CSR_colIdx.txt', 'w') as f:
        for item in CSR_colIdx:
            f.write(f"{item}\n")
        # f.write(f"{CSR_colIdx}")
    
    with open('matrix_data/CSR_rowPtr.txt', 'w') as f:
        for item in CSR_rowPtr:
            f.write(f"{item}\n")
        # f.write(f"{CSR_rowPtr}")
    
    with open('matrix_data/CSR_redColIdx.txt', 'w') as f:
        for item in CSR_redColIdx:
            f.write(f"{item}\n")
        # f.write(f"{CSR_redColIdx}")
            
    with open('matrix_data/GeneralInfo.txt', 'w') as f:
        for item in [no_of_rows, no_of_cols, NNZ_values, max(CSR_redColIdx)+1]:
            f.write(f"{item}\n")

Main(11)