#include "../include/permutation.hpp"
#include "../include/disk_to_memory.hpp"
#include "../include/csr.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>

using namespace std;

const double FLOAT_TOL = 1e-5;

/*
  Test: Permute X and unpermute X - should get back original X
 */
bool test_permute_unpermute_X(const string& x_filename) {
    cout << "\n" << string(60, '=') << endl;
    cout << "Testing: " << x_filename << endl;
    cout << string(60, '=') << endl;
    
    try {
        // Load X
        string x_path = "../dataset/X/" + x_filename;
        cout << "Loading X from: " << x_path << endl;
        CSR X = load_X_h5_as_csr(x_path, "");
        
        cout << "Original X:" << endl;
        cout << "  Rows: " << X.nrows << ", Cols: " << X.ncols << ", nnz: " << X.nnz << endl;
        
        // Compute nnz per row
        vector<size_t> nnz_per_row = compute_nnz_per_row(X);
        
        // Create row permutation (new2old mapping)
        vector<int> row_new2old = create_row_new2old(nnz_per_row, true);
        cout << "  Created row permutation (size: " << row_new2old.size() << ")" << endl;
        
        // Permute X
        cout << "Permuting X..." << endl;
        CSR X_permuted = permute_csr_rows(X, row_new2old);
        cout << "Permuted X:" << endl;
        cout << "  Rows: " << X_permuted.nrows << ", Cols: " << X_permuted.ncols << ", nnz: " << X_permuted.nnz << endl;
        
        // Verify dimensions match
        if (X_permuted.nrows != X.nrows || X_permuted.ncols != X.ncols || X_permuted.nnz != X.nnz) {
            cout << "  ✗ Dimension mismatch!" << endl;
            return false;
        }
        cout << "  ✓ Dimensions match" << endl;
        
        // Unpermute X: Use direct unpermute function
        cout << "Unpermuting X..." << endl;
        CSR X_recovered = unpermute_csr_rows(X_permuted, row_new2old);
        cout << "Recovered X:" << endl;
        cout << "  Rows: " << X_recovered.nrows << ", Cols: " << X_recovered.ncols << ", nnz: " << X_recovered.nnz << endl;
        
        // Verify X_recovered matches X
        bool match = true;
        if (X_recovered.nrows != X.nrows || X_recovered.ncols != X.ncols || X_recovered.nnz != X.nnz) {
            cout << "  ✗ Dimension mismatch!" << endl;
            return false;
        }
        
        // Compare row by row
        int mismatched_rows = 0;
        int total_mismatches = 0;
        for (int i = 0; i < X.nrows; i++) {
            int start_X = X.indptr[i];
            int end_X = X.indptr[i + 1];
            int start_Xr = X_recovered.indptr[i];
            int end_Xr = X_recovered.indptr[i + 1];
            
            if (end_X - start_X != end_Xr - start_Xr) {
                if (mismatched_rows < 5) {
                    cout << "  ✗ Row " << i << " nnz mismatch: " << (end_X - start_X) << " vs " << (end_Xr - start_Xr) << endl;
                }
                mismatched_rows++;
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
                    if (total_mismatches < 5) {
                        cout << "  ✗ Row " << i << ", nonzero " << idx << " mismatch: (" << j_X << ", " << val_X 
                             << ") vs (" << j_Xr << ", " << val_Xr << ")" << endl;
                    }
                    total_mismatches++;
                    match = false;
                }
            }
        }
        
        if (match) {
            cout << "  ✓ X_recovered matches X (permute + unpermute = identity)" << endl;
            return true;
        } else {
            cout << "  ✗ X_recovered does not match X" << endl;
            cout << "    Mismatched rows: " << mismatched_rows << endl;
            cout << "    Total mismatches: " << total_mismatches << endl;
            return false;
        }
        
    } catch (const exception& e) {
        cerr << "  ✗ Error: " << e.what() << endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    cout << "=== Permute/Unpermute X Test for Multiple Datasets ===" << endl;
    cout << "Testing: permute X → unpermute X → should equal original X" << endl;
    cout << "Note: Only permuting X (not W)" << endl;
    
    vector<string> datasets = {"d0.h5", "d2.h5", "d3.h5", "d4.h5", "d5.h5"};
    
    if (argc > 1) {
        // If datasets provided as arguments, use those
        datasets.clear();
        for (int i = 1; i < argc; i++) {
            datasets.push_back(argv[i]);
        }
    }
    
    vector<bool> results;
    int passed = 0;
    int failed = 0;
    
    for (const auto& dataset : datasets) {
        bool result = test_permute_unpermute_X(dataset);
        results.push_back(result);
        if (result) {
            passed++;
        } else {
            failed++;
        }
    }
    
    cout << "\n" << string(60, '=') << endl;
    cout << "=== Test Summary ===" << endl;
    cout << string(60, '=') << endl;
    for (size_t i = 0; i < datasets.size(); i++) {
        cout << datasets[i] << ": " << (results[i] ? "PASSED ✓" : "FAILED ✗") << endl;
    }
    cout << "\nTotal: " << passed << " passed, " << failed << " failed" << endl;
    
    if (failed == 0) {
        cout << "\n✓ All tests passed!" << endl;
        return 0;
    } else {
        cout << "\n✗ Some tests failed!" << endl;
        return 1;
    }
}

