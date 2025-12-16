# Sparse Matrix Multiplication Baseline

This directory contains a simple baseline implementation for sparse-dense matrix multiplication: `Y = X * W`

## Structure

- **`csr.hpp`**: CSR (Compressed Sparse Row) matrix data structure
- **`disk_to_memory.hpp/cpp`**: Functions to load X and W from HDF5 files
- **`spmm.hpp`**: Sparse-dense matrix multiplication implementation
- **`main.cpp`**: Main program that orchestrates the computation

## Input Files

- **X**: Sparse matrix from `dataset/X/d0.h5` (10x Genomics HDF5 format)
- **W**: Dense matrix from `dataset/W/w0.h5`

## Output

- **Y**: Result matrix saved to `dataset/Y/y0.h5`

## Building

### Option 1: Using build script (PowerShell)
```powershell
cd include
.\build.ps1
```

### Option 2: Manual compilation

**In Git Bash:**
```bash
g++ -std=c++17 -O3 -Wall main.cpp disk_to_memory.cpp -o run.exe $(pkg-config --cflags --libs hdf5) -lhdf5_cpp
```

**In PowerShell:**
```powershell
$hdf5 = pkg-config --cflags --libs hdf5
g++ -std=c++17 -O3 -Wall main.cpp disk_to_memory.cpp -o run.exe $hdf5.Split() -lhdf5_cpp
```

**Note**: This uses `pkg-config` to find HDF5, and adds `-lhdf5_cpp` for the C++ API.

## Running

```bash
cd include
.\run.exe d0.h5 w0.h5
```

**Arguments:**
- First argument: X filename (e.g., `d0.h5`, `d1.h5`)
- Second argument: W filename (e.g., `w0.h5`, `w1.h5`)

The output filename is automatically generated based on the postfix of the X file:
- `d0.h5` + `w0.h5` → `y0.h5`
- `d1.h5` + `w1.h5` → `y1.h5`

**Examples:**
```bash
.\run.exe d0.h5 w0.h5  # Output: dataset/Y/y0.h5
.\run.exe d1.h5 w1.h5  # Output: dataset/Y/y1.h5
.\run.exe d2.h5 w2.h5  # Output: dataset/Y/y2.h5
```

The program will:
1. Load X from `dataset/X/<X_file>` (converting from CSC to CSR format)
2. Load W from `dataset/W/<W_file>`
3. Compute Y = X * W
4. Save Y to `dataset/Y/y<postfix>.h5`
5. Print timing information

## Implementation Details

### CSC to CSR Conversion
The 10x Genomics format stores sparse matrices in CSC (Compressed Sparse Column) format. We convert this to CSR (Compressed Sparse Row) format for efficient row-wise operations during SpMM.

### SpMM Algorithm
The baseline uses a simple row-wise algorithm:
```
for each row i in X:
    for each non-zero X[i,k]:
        for each column j in W:
            Y[i,j] += X[i,k] * W[k,j]
```

Time complexity: O(nnz_X * k) where k is the number of columns in W.

