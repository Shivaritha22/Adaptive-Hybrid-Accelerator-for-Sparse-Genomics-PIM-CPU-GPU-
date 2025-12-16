#include "../include/permutation.hpp"
#include "../include/csr.hpp"
#include "../include/spmm.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace std;

const double FLOAT_TOL = 1e-5;

/*
  Test: Permute X and unpermute X - should get back original X
 */
bool test_permute_unpermute_X() {
    cout << "\n=== Test: Permute X and Unpermute X ===" << endl;
    
    // Create a small test CSR matrix (4x4)
    CSR X;
    X.nrows = 4;
    X.ncols = 4;
    X.nnz = 6;
    X.indptr = {0, 2, 3, 5, 6};
    X.indices = {0, 2,    1,    0, 2,    3};
    X.data = {1.0f, 2.0f,  3.0f,  4.0f, 5.0f,  6.0f};
    
    cout << "Original X:" << endl;
    cout << "  Rows: " << X.nrows << ", Cols: " << X.ncols << ", nnz: " << X.nnz << endl;
    
    // Compute nnz per row
    vector<size_t> nnz_per_row = compute_nnz_per_row(X);
    cout << "  nnz_per_row: ";
    for (size_t i = 0; i < nnz_per_row.size(); i++) {
        cout << nnz_per_row[i] << " ";
    }
    cout << endl;
    
    // Create row permutation (new2old mapping)
    vector<int> row_new2old = create_row_new2old(nnz_per_row, true);
    cout << "  row_new2old: ";
    for (size_t i = 0; i < row_new2old.size(); i++) {
        cout << row_new2old[i] << " ";
    }
    cout << endl;
    
    // Permute X
    CSR X_permuted = permute_csr_rows(X, row_new2old);
    cout << "Permuted X:" << endl;
    cout << "  Rows: " << X_permuted.nrows << ", Cols: " << X_permuted.ncols << ", nnz: " << X_permuted.nnz << endl;
    
    // Verify dimensions match
    assert(X_permuted.nrows == X.nrows);
    assert(X_permuted.ncols == X.ncols);
    assert(X_permuted.nnz == X.nnz);
    cout << "  ✓ Dimensions match" << endl;
    
    // Unpermute X: Create inverse mapping
    // row_new2old[new] = old means: X'[new] = X[old]
    // To recover X: X[old] = X'[new] where new = row_old2new[old]
    // row_new2old_inv where row_new2old_inv[new] = old for the inverse permutation
    // But actually, row_new2old_inv[old] = new, then use it as new2old
    
    // Step 1: Create old2new mapping
    vector<int> row_old2new(X.nrows);
    for (int i_new = 0; i_new < X.nrows; i_new++) {
        int i_old = row_new2old[i_new];
        row_old2new[i_old] = i_new;  // old → new
    }
    
    // Step 2: Create inverse new2old mapping for unpermutation
    // row_new2old_inv[new] = old, where new comes from row_old2new[old]
    vector<int> row_new2old_inv(X.nrows);
    for (int i_old = 0; i_old < X.nrows; i_old++) {
        int i_new = row_old2new[i_old];
        row_new2old_inv[i_new] = i_old;  // new → old (inverse)
    }
    
    // Step 3: Permute X_permuted with inverse mapping to recover X
    CSR X_recovered = permute_csr_rows(X_permuted, row_new2old_inv);
    
    // Verify X_recovered matches X
    bool match = true;
    if (X_recovered.nrows != X.nrows || X_recovered.ncols != X.ncols || X_recovered.nnz != X.nnz) {
        cout << "  ✗ Dimension mismatch!" << endl;
        match = false;
    }
    
    // Compare row by row
    for (int i = 0; i < X.nrows; i++) {
        int start_X = X.indptr[i];
        int end_X = X.indptr[i + 1];
        int start_Xr = X_recovered.indptr[i];
        int end_Xr = X_recovered.indptr[i + 1];
        
        if (end_X - start_X != end_Xr - start_Xr) {
            cout << "  ✗ Row " << i << " nnz mismatch: " << (end_X - start_X) << " vs " << (end_Xr - start_Xr) << endl;
            match = false;
            continue;
        }
        
        // Compare nonzeros
        for (int idx = 0; idx < end_X - start_X; idx++) {
            int j_X = X.indices[start_X + idx];
            float val_X = X.data[start_X + idx];
            int j_Xr = X_recovered.indices[start_Xr + idx];
            float val_Xr = X_recovered.data[start_Xr + idx];
            
            if (j_X != j_Xr || fabs(val_X - val_Xr) > FLOAT_TOL) {
                cout << "  ✗ Row " << i << ", nonzero " << idx << " mismatch: (" << j_X << ", " << val_X 
                     << ") vs (" << j_Xr << ", " << val_Xr << ")" << endl;
                match = false;
            }
        }
    }
    
    if (match) {
        cout << "  ✓ X_recovered matches X (permute + unpermute = identity)" << endl;
        return true;
    } else {
        cout << "  ✗ X_recovered does not match X" << endl;
        return false;
    }
}

/*
  Test: Permute W and unpermute W - should get back original W
 */
bool test_permute_unpermute_W() {
    cout << "\n=== Test: Permute W and Unpermute W ===" << endl;
    
    // Create a small test weight matrix W (4x2)
    int W_rows = 4;
    int W_cols = 2;
    vector<float> W = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };
    
    cout << "Original W:" << endl;
    cout << "  Rows: " << W_rows << ", Cols: " << W_cols << endl;
    
    // Create a simple row permutation 
    vector<int> row_new2old = {2, 0, 3, 1};
    cout << "  row_new2old: ";
    for (size_t i = 0; i < row_new2old.size(); i++) {
        cout << row_new2old[i] << " ";
    }
    cout << endl;
    
    // Permute W
    vector<float> W_permuted = permute_weight_rows(W, W_rows, W_cols, row_new2old);
    cout << "Permuted W:" << endl;
    cout << "  W_permuted values: ";
    for (size_t i = 0; i < W_permuted.size(); i++) {
        cout << W_permuted[i] << " ";
    }
    cout << endl;
    
    vector<float> W_recovered = unpermute_rows(W_permuted, W_rows, W_cols, row_new2old);
    
    cout << "  W_recovered values: ";
    for (size_t i = 0; i < W_recovered.size(); i++) {
        cout << W_recovered[i] << " ";
    }
    cout << endl;
    
    // Verify W_recovered matches W
    bool match = true;
    if (W_recovered.size() != W.size()) {
        cout << "  ✗ Size mismatch!" << endl;
        match = false;
    }
    
    for (size_t i = 0; i < W.size(); i++) {
        if (fabs(W_recovered[i] - W[i]) > FLOAT_TOL) {
            cout << "  ✗ Mismatch at index " << i << ": " << W_recovered[i] << " vs " << W[i] << endl;
            match = false;
        }
    }
    
    if (match) {
        cout << "  ✓ W_recovered matches W (permute + unpermute = identity)" << endl;
        return true;
    } else {
        cout << "  ✗ W_recovered does not match W" << endl;
        return false;
    }
}

int main() {
    cout << "=== Simple Permutation Tests (Row Only) ===" << endl;
    
    bool test1_ok = test_permute_unpermute_X();
    bool test2_ok = test_permute_unpermute_W();
    
    cout << "\n=== Test Summary ===" << endl;
    cout << "Test 1 (Permute/Unpermute X): " << (test1_ok ? "PASSED" : "FAILED") << endl;
    cout << "Test 2 (Permute/Unpermute W): " << (test2_ok ? "PASSED" : "FAILED") << endl;
    
    if (test1_ok && test2_ok) {
        cout << "\n✓ All tests passed!" << endl;
        return 0;
    } else {
        cout << "\n✗ Some tests failed!" << endl;
        return 1;
    }
}

