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

/**
 * Compare two CSR matrices for equality
 */
bool compare_csr_matrices(const CSR& X1, const CSR& X2, const string& name1, const string& name2) {
    if (X1.nrows != X2.nrows || X1.ncols != X2.ncols || X1.nnz != X2.nnz) {
        cout << "  ✗ Dimension mismatch: " << name1 << " (" << X1.nrows << "x" << X1.ncols 
             << ", nnz=" << X1.nnz << ") vs " << name2 << " (" << X2.nrows << "x" << X2.ncols 
             << ", nnz=" << X2.nnz << ")" << endl;
        return false;
    }
    
    int mismatched_rows = 0;
    int total_mismatches = 0;
    
    for (int i = 0; i < X1.nrows; i++) {
        int start_X1 = X1.indptr[i];
        int end_X1 = X1.indptr[i + 1];
        int start_X2 = X2.indptr[i];
        int end_X2 = X2.indptr[i + 1];
        
        if (end_X1 - start_X1 != end_X2 - start_X2) {
            if (mismatched_rows < 5) {
                cout << "  ✗ Row " << i << " nnz mismatch: " << (end_X1 - start_X1) 
                     << " vs " << (end_X2 - start_X2) << endl;
            }
            mismatched_rows++;
            continue;
        }
        
        // Compare nonzeros
        for (int idx = 0; idx < end_X1 - start_X1; idx++) {
            int j_X1 = X1.indices[start_X1 + idx];
            float val_X1 = X1.data[start_X1 + idx];
            int j_X2 = X2.indices[start_X2 + idx];
            float val_X2 = X2.data[start_X2 + idx];
            
            if (j_X1 != j_X2 || fabs(val_X1 - val_X2) > FLOAT_TOL) {
                if (total_mismatches < 5) {
                    cout << "  ✗ Row " << i << ", nonzero " << idx << " mismatch: (" 
                         << j_X1 << ", " << val_X1 << ") vs (" << j_X2 << ", " << val_X2 << ")" << endl;
                }
                total_mismatches++;
            }
        }
    }
    
    if (mismatched_rows == 0 && total_mismatches == 0) {
        cout << "  ✓ " << name1 << " matches " << name2 << endl;
        return true;
    } else {
        cout << "  ✗ " << name1 << " does not match " << name2 << endl;
        cout << "    Mismatched rows: " << mismatched_rows << endl;
        cout << "    Total mismatches: " << total_mismatches << endl;
        return false;
    }
}

/**
 * Compare two weight matrices for equality
 */
bool compare_weight_matrices(const vector<float>& W1, const vector<float>& W2, 
                             int W_rows, int W_cols, 
                             const string& name1, const string& name2) {
    if (static_cast<int>(W1.size()) != W_rows * W_cols || 
        static_cast<int>(W2.size()) != W_rows * W_cols) {
        cout << "  ✗ Size mismatch: " << name1 << " (" << W1.size() 
             << ") vs " << name2 << " (" << W2.size() << ")" << endl;
        return false;
    }
    
    int mismatches = 0;
    for (int i = 0; i < W_rows * W_cols; ++i) {
        if (fabs(W1[i] - W2[i]) > FLOAT_TOL) {
            if (mismatches < 5) {
                int row = i / W_cols;
                int col = i % W_cols;
                cout << "  ✗ Element [" << row << ", " << col << "] mismatch: " 
                     << W1[i] << " vs " << W2[i] << endl;
            }
            mismatches++;
        }
    }
    
    if (mismatches == 0) {
        cout << "  ✓ " << name1 << " matches " << name2 << endl;
        return true;
    } else {
        cout << "  ✗ " << name1 << " does not match " << name2 << endl;
        cout << "    Total mismatches: " << mismatches << endl;
        return false;
    }
}

/**
 * Test function for row+col permutation workflow:
 * Step 1: Permute row X
 * Step 2: Permute col X and row W (these two permutations must match)
 * Step 3: Unpermute col of X, then unpermute row of X. Check if the resultant X is the same.
 *         The order is very important. First col then row.
 * Step 4: Unpermute row W. Check with original value
 */
bool test_permute_row_col_workflow(const string& x_filename, const string& w_filename) {
    cout << "\n" << string(60, '=') << endl;
    cout << "Testing Row+Col Permutation Workflow" << endl;
    cout << "X: " << x_filename << ", W: " << w_filename << endl;
    cout << string(60, '=') << endl;
    
    try {
        // Load X and W
        string x_path = "../dataset/X/" + x_filename;
        string w_path = "../dataset/W/" + w_filename;
        
        cout << "\nLoading matrices..." << endl;
        cout << "  X from: " << x_path << endl;
        CSR X_original = load_X_h5_as_csr(x_path, "");
        cout << "  X: Rows=" << X_original.nrows << ", Cols=" << X_original.ncols 
             << ", nnz=" << X_original.nnz << endl;
        
        cout << "  W from: " << w_path << endl;
        int W_rows, W_cols;
        vector<float> W_original = load_W_h5(w_path, W_rows, W_cols, "");
        cout << "  W: Rows=" << W_rows << ", Cols=" << W_cols << endl;
        
        // Verify dimensions match for matrix multiplication
        if (X_original.ncols != W_rows) {
            cout << "  ✗ Dimension mismatch: X.ncols (" << X_original.ncols 
                 << ") != W.rows (" << W_rows << ")" << endl;
            return false;
        }
        cout << "  ✓ Dimensions compatible for X * W" << endl;
        
        // ============================================================
        // Step 1: Permute row X
        // ============================================================
        cout << "\n" << string(60, '-') << endl;
        cout << "Step 1: Permute row X" << endl;
        cout << string(60, '-') << endl;
        
        vector<size_t> nnz_per_row = compute_nnz_per_row(X_original);
        vector<int> row_new2old_X = create_row_new2old(nnz_per_row, true);
        cout << "  Created row permutation for X (size: " << row_new2old_X.size() << ")" << endl;
        
        CSR X_row_permuted = permute_csr_rows(X_original, row_new2old_X);
        cout << "  Permuted X rows: Rows=" << X_row_permuted.nrows 
             << ", Cols=" << X_row_permuted.ncols << ", nnz=" << X_row_permuted.nnz << endl;
        
        // ============================================================
        // Step 2: Permute col X and row W (these two permutations must match)
        // ============================================================
        cout << "\n" << string(60, '-') << endl;
        cout << "Step 2: Permute col X and row W (same permutation)" << endl;
        cout << string(60, '-') << endl;
        
        vector<size_t> nnz_per_col = compute_nnz_per_col(X_row_permuted);
        vector<int> col_new2old = create_col_new2old(nnz_per_col, true);
        cout << "  Created column permutation for X (size: " << col_new2old.size() << ")" << endl;
        
        // Verify that col permutation size matches W rows
        if (static_cast<int>(col_new2old.size()) != W_rows) {
            cout << "  ✗ Column permutation size (" << col_new2old.size() 
                 << ") != W.rows (" << W_rows << ")" << endl;
            return false;
        }
        
        // Permute columns of X
        CSR X_row_col_permuted = permute_csr_cols(X_row_permuted, col_new2old);
        cout << "  Permuted X columns: Rows=" << X_row_col_permuted.nrows 
             << ", Cols=" << X_row_col_permuted.ncols << ", nnz=" << X_row_col_permuted.nnz << endl;
        
        // Permute rows of W using the same permutation (col_new2old)
        // Note: col_new2old is for X columns, which corresponds to W rows
        vector<float> W_row_permuted = permute_weight_rows(W_original, W_rows, W_cols, col_new2old);
        cout << "  Permuted W rows: Rows=" << W_rows << ", Cols=" << W_cols << endl;
        
        // ============================================================
        // Step 3: Unpermute col of X, then unpermute row of X
        // Order is very important: First col then row
        // ============================================================
        cout << "\n" << string(60, '-') << endl;
        cout << "Step 3: Unpermute col of X, then unpermute row of X" << endl;
        cout << "  Order: First col, then row" << endl;
        cout << string(60, '-') << endl;
        
        // First: Unpermute columns
        CSR X_row_only = unpermute_csr_cols(X_row_col_permuted, col_new2old);
        cout << "  After unpermuting columns: Rows=" << X_row_only.nrows 
             << ", Cols=" << X_row_only.ncols << ", nnz=" << X_row_only.nnz << endl;
        
        // Second: Unpermute rows
        CSR X_recovered = unpermute_csr_rows(X_row_only, row_new2old_X);
        cout << "  After unpermuting rows: Rows=" << X_recovered.nrows 
             << ", Cols=" << X_recovered.ncols << ", nnz=" << X_recovered.nnz << endl;
        
        // Check if X_recovered matches X_original
        cout << "\n  Checking if X_recovered matches X_original..." << endl;
        bool X_match = compare_csr_matrices(X_recovered, X_original, "X_recovered", "X_original");
        
        // ============================================================
        // Step 4: Unpermute row W. Check with original value
        // ============================================================
        cout << "\n" << string(60, '-') << endl;
        cout << "Step 4: Unpermute row W" << endl;
        cout << string(60, '-') << endl;
        
        vector<float> W_recovered = unpermute_rows(W_row_permuted, W_rows, W_cols, col_new2old);
        cout << "  Unpermuted W: Rows=" << W_rows << ", Cols=" << W_cols << endl;
        
        // Check if W_recovered matches W_original
        cout << "\n  Checking if W_recovered matches W_original..." << endl;
        bool W_match = compare_weight_matrices(W_recovered, W_original, W_rows, W_cols, 
                                               "W_recovered", "W_original");
        
        // ============================================================
        // Summary
        // ============================================================
        cout << "\n" << string(60, '=') << endl;
        cout << "Test Summary" << endl;
        cout << string(60, '=') << endl;
        cout << "  Step 3 (X recovery): " << (X_match ? "PASSED ✓" : "FAILED ✗") << endl;
        cout << "  Step 4 (W recovery): " << (W_match ? "PASSED ✓" : "FAILED ✗") << endl;
        
        bool all_passed = X_match && W_match;
        if (all_passed) {
            cout << "\n✓ All steps passed!" << endl;
        } else {
            cout << "\n✗ Some steps failed!" << endl;
        }
        
        return all_passed;
        
    } catch (const exception& e) {
        cerr << "  ✗ Error: " << e.what() << endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    cout << "=== Row+Col Permutation Workflow Test ===" << endl;
    cout << "Tests the complete permutation workflow:" << endl;
    cout << "  1. Permute row X" << endl;
    cout << "  2. Permute col X and row W (same permutation)" << endl;
    cout << "  3. Unpermute col X, then unpermute row X (check = original)" << endl;
    cout << "  4. Unpermute row W (check = original)" << endl;
    cout << endl;
    
     // Define test cases: (X_file, W_file)
     vector<pair<string, string>> test_cases = {
        {"d0.h5", "w0.h5"},
        {"d2.h5", "w2.h5"},
        {"d3.h5", "w3.h5"},
        {"d4.h5", "w4.h5"},
        {"d5.h5", "w5.h5"}
    };
    
    // If command line arguments provided, use those instead
    if (argc >= 3) {
        test_cases.clear();
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 < argc) {
                test_cases.push_back({argv[i], argv[i + 1]});
            }
        }
    }
    vector<bool> results;
    int passed = 0;
    int failed = 0;
    
    for (size_t i = 0; i < test_cases.size(); ++i) {
        cout << "\n" << string(80, '=') << endl;
        cout << "TEST CASE " << (i + 1) << ": " << test_cases[i].first 
             << " + " << test_cases[i].second << endl;
        cout << string(80, '=') << endl;
        
        bool result = test_permute_row_col_workflow(test_cases[i].first, test_cases[i].second);
        results.push_back(result);
        
        if (result) {
            passed++;
        } else {
            failed++;
        }
    }
    
    // Final summary
    cout << "\n" << string(80, '=') << endl;
    cout << "=== FINAL TEST SUMMARY ===" << endl;
    cout << string(80, '=') << endl;
    for (size_t i = 0; i < test_cases.size(); ++i) {
        cout << "Test Case " << (i + 1) << " (" << test_cases[i].first 
             << " + " << test_cases[i].second << "): " 
             << (results[i] ? "PASSED ✓" : "FAILED ✗") << endl;
    }
    cout << "\nTotal: " << passed << " passed, " << failed << " failed" << endl;
    if (failed == 0) {
        cout << "\n✓ All test cases passed!" << endl;
        return 0;
    } else {
        cout << "\n✗ Some test cases failed!" << endl;
        return 1;
    }
    
    
}
