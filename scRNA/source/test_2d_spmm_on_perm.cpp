#include "../include/permutation.hpp"
#include "../include/disk_to_memory.hpp"
#include "../include/spmm.hpp"
#include "../include/csr.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <H5Cpp.h>

using namespace std;
using namespace H5;

const double FLOAT_TOL = 1e-5;

string extract_postfix(const string& filename) {
    // Remove .h5 extension
    string name = filename;
    size_t dot_pos = name.find_last_of('.');
    if (dot_pos != string::npos) {
        name = name.substr(0, dot_pos);
    }
    
    // Extract the numeric postfix
    if (name.length() > 1) {
        return name.substr(1); 
    }
    return "0"; // Fallback
}


vector<float> load_Y_h5(const string& y_h5_path, int& nrows, int& ncols) {
    vector<float> Y_data;
    
    try {
        H5File file(y_h5_path, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet("Y");
        DataSpace dataspace = dataset.getSpace();
        
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        nrows = dims[0];
        ncols = dims[1];
        
        Y_data.resize(nrows * ncols);
        dataset.read(Y_data.data(), PredType::NATIVE_FLOAT);
        
    } catch (Exception& e) {
        cerr << "HDF5 error loading Y: " << e.getDetailMsg() << endl;
        throw;
    }
    
    return Y_data;
}

/*
  Compare two result matrices and compute error metrics
  Compares Y_final (result from permutation + SpMM + unpermutation) against Y_baseline
  (result from baseline run). This verifies that permutation + SpMM + unpermutation
  produces mathematically equivalent results.
*/
bool compare_results(const vector<float>& Y_final, const vector<float>& Y_baseline, 
                    int rows, int cols, const string& test_name) {
    cout << "\n" << string(60, '=') << endl;
    cout << "COMPARISON: " << test_name << endl;
    cout << string(60, '=') << endl;
    
    // Validate dimensions
    if (Y_final.size() != Y_baseline.size()) {
        cerr << "ERROR: Size mismatch!" << endl;
        cerr << "  Y_final size: " << Y_final.size() << endl;
        cerr << "  Y_baseline size: " << Y_baseline.size() << endl;
        cerr << "  Expected: " << (rows * cols) << " (rows=" << rows << ", cols=" << cols << ")" << endl;
        return false;
    }
    
    if (Y_final.size() != static_cast<size_t>(rows * cols)) {
        cerr << "ERROR: Size doesn't match dimensions!" << endl;
        cerr << "  Actual size: " << Y_final.size() << endl;
        cerr << "  Expected: " << (rows * cols) << " (rows=" << rows << ", cols=" << cols << ")" << endl;
        return false;
    }
    
    // Compute error statistics
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double sum_abs_error = 0.0;
    double sum_sq_error = 0.0;
    size_t max_abs_error_idx = 0;
    size_t max_rel_error_idx = 0;
    size_t num_nonzero_baseline = 0;
    
    for (size_t i = 0; i < Y_final.size(); i++) {
        double abs_error = fabs(Y_final[i] - Y_baseline[i]);
        double rel_error = 0.0;
        
        sum_abs_error += abs_error;
        sum_sq_error += abs_error * abs_error;
        
        if (fabs(Y_baseline[i]) > 1e-10) {
            rel_error = abs_error / fabs(Y_baseline[i]);
            num_nonzero_baseline++;
        } else if (abs_error > 1e-10) {
            rel_error = abs_error;  // For near-zero baseline, use absolute error
        }
        
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_abs_error_idx = i;
        }
        
        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
            max_rel_error_idx = i;
        }
    }
    
    double mean_abs_error = sum_abs_error / Y_final.size();
    double rmse = sqrt(sum_sq_error / Y_final.size());
    
    // Print statistics
    cout << fixed << setprecision(10);
    cout << "Max absolute error: " << max_abs_error << " (at index " << max_abs_error_idx << ")" << endl;
    cout << "Max relative error: " << max_rel_error << " (at index " << max_rel_error_idx << ")" << endl;
    cout << "Mean absolute error: " << mean_abs_error << endl;
    cout << "RMSE: " << rmse << endl;
    cout << "Non-zero baseline elements: " << num_nonzero_baseline << " / " << Y_final.size() << endl;
    
    // Check if within tolerance
    bool passed = (max_abs_error < FLOAT_TOL);
    
    if (passed) {
        cout << "\n✓ PASSED: Max absolute error (" << max_abs_error << ") < tolerance (" << FLOAT_TOL << ")" << endl;
    } else {
        cout << "\n✗ FAILED: Max absolute error (" << max_abs_error << ") >= tolerance (" << FLOAT_TOL << ")" << endl;
        int row = max_abs_error_idx / cols;
        int col = max_abs_error_idx % cols;
        cout << "  Location of max error: row=" << row << ", col=" << col << endl;
        cout << "  Y_final[" << row << ", " << col << "] = " << Y_final[max_abs_error_idx] << endl;
        cout << "  Y_baseline[" << row << ", " << col << "] = " << Y_baseline[max_abs_error_idx] << endl;
    }
    
    return passed;
}

/*
  Test function for 2D SpMM on permuted matrices:
  Step 1: Permute row X
  Step 2: Permute col X and row W (these two permutations must match)
  Step 3: Perform Y = X * W on the permuted values
  Step 4: Unpermute ONLY row of Y
  Step 5: Compare this Y output to the baseline's Y outputs
*/
bool test_2d_spmm_on_perm(const string& x_filename, const string& w_filename) {
    cout << "\n" << string(60, '=') << endl;
    cout << "Testing 2D SpMM on Permuted Matrices" << endl;
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
        vector<int> row_new2old = create_row_new2old(nnz_per_row, true);
        cout << "  Created row permutation for X (size: " << row_new2old.size() << ")" << endl;
        
        CSR X_row_permuted = permute_csr_rows(X_original, row_new2old);
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
        // Step 3: Perform Y = X * W on the permuted values
        // ============================================================
        cout << "\n" << string(60, '-') << endl;
        cout << "Step 3: Perform Y = X * W on permuted matrices" << endl;
        cout << string(60, '-') << endl;
        
        vector<float> Y_permuted = spmm_baseline(X_row_col_permuted, W_row_permuted, W_rows, W_cols);
        int Y_rows = X_row_col_permuted.nrows;
        int Y_cols = W_cols;
        cout << "  Y_permuted: Rows=" << Y_rows << ", Cols=" << Y_cols << endl;
        
        // ============================================================
        // Step 4: Unpermute ONLY row of Y
        // ============================================================
        cout << "\n" << string(60, '-') << endl;
        cout << "Step 4: Unpermute ONLY row of Y" << endl;
        cout << string(60, '-') << endl;
        
        vector<float> Y_final = unpermute_rows(Y_permuted, Y_rows, Y_cols, row_new2old);
        cout << "  Y_final: Rows=" << Y_rows << ", Cols=" << Y_cols << endl;
        
        // ============================================================
        // Step 5: Compare this Y output to the baseline's Y outputs
        // ============================================================
        cout << "\n" << string(60, '-') << endl;
        cout << "Step 5: Compare Y_final with baseline Y" << endl;
        cout << string(60, '-') << endl;
        
        // Extract postfix from x_filename (e.g., "d0.h5" -> "0")
        string postfix = extract_postfix(x_filename);
        string y_baseline_path = "../dataset/Y/y" + postfix + ".h5";
        cout << "  Loading baseline Y from: " << y_baseline_path << endl;
        
        int y_baseline_rows, y_baseline_cols;
        vector<float> Y_baseline = load_Y_h5(y_baseline_path, y_baseline_rows, y_baseline_cols);
        cout << "  Y_baseline: Rows=" << y_baseline_rows << ", Cols=" << y_baseline_cols << endl;
        
        // Verify dimensions match
        if (Y_rows != y_baseline_rows || Y_cols != y_baseline_cols) {
            cout << "  ✗ Dimension mismatch: Y_final (" << Y_rows << "x" << Y_cols 
                 << ") vs Y_baseline (" << y_baseline_rows << "x" << y_baseline_cols << ")" << endl;
            return false;
        }
        
        // Compare results
        bool match = compare_results(Y_final, Y_baseline, Y_rows, Y_cols, 
                                    "Y_final vs Y_baseline");
        
        // ============================================================
        // Summary
        // ============================================================
        cout << "\n" << string(60, '=') << endl;
        cout << "Test Summary" << endl;
        cout << string(60, '=') << endl;
        cout << "  Step 5 (Y comparison): " << (match ? "PASSED ✓" : "FAILED ✗") << endl;
        
        if (match) {
            cout << "\n✓ All steps passed!" << endl;
        } else {
            cout << "\n✗ Test failed!" << endl;
        }
        
        return match;
        
    } catch (const exception& e) {
        cerr << "  ✗ Error: " << e.what() << endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    cout << "=== 2D SpMM on Permuted Matrices Test ===" << endl;
    cout << "Tests the complete permutation + SpMM workflow:" << endl;
    cout << "  1. Permute row X" << endl;
    cout << "  2. Permute col X and row W (same permutation)" << endl;
    cout << "  3. Perform Y = X * W on permuted matrices" << endl;
    cout << "  4. Unpermute ONLY row of Y" << endl;
    cout << "  5. Compare Y output to baseline Y outputs" << endl;
    cout << endl;
    
    // Define test cases: (X_file, W_file)
    vector<pair<string, string>> test_cases = {
        {"d0.h5", "w0.h5"},
        {"d2.h5", "w2.h5"},
        {"d3.h5", "w3.h5"},
        {"d4.h5", "w4.h5"},
        {"d5.h5", "w5.h5"}
    };
    
    
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
        
        bool result = test_2d_spmm_on_perm(test_cases[i].first, test_cases[i].second);
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

