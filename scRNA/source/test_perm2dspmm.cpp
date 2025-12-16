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
#include <filesystem>

using namespace std;
using namespace H5;
namespace fs = std::filesystem;

const double ABS_TOL = 1e-4;   
const double REL_TOL = 1e-5;   

bool approx_equal(float a, float b) {
    float diff  = fabs(a - b);
    float maxab = fmax(fabs(a), fabs(b));
    return diff <= ABS_TOL || diff <= REL_TOL * maxab;
}


/**
 * Extract postfix from filename (e.g., "d0.h5" -> "0")
 */
string extract_postfix(const string& filename) {
    string name = filename;
    size_t dot_pos = name.find_last_of('.');
    if (dot_pos != string::npos) {
        name = name.substr(0, dot_pos);
    }
    if (name.length() > 1) {
        return name.substr(1);
    }
    return "0";
}

/**
 * Load Y matrix from HDF5 file
 */
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

/**
 * Save Y matrix to HDF5 file
 */
void save_Y_h5(const vector<float>& Y, int rows, int cols, const string& path) {
    try {
        fs::create_directories(fs::path(path).parent_path());
        H5File file(path, H5F_ACC_TRUNC);
        hsize_t dims[2] = {static_cast<hsize_t>(rows), static_cast<hsize_t>(cols)};
        DataSpace dataspace(2, dims);
        DataSet dataset = file.createDataSet("Y", PredType::NATIVE_FLOAT, dataspace);
        dataset.write(Y.data(), PredType::NATIVE_FLOAT);
    } catch (Exception& e) {
        cerr << "HDF5 error saving Y: " << e.getDetailMsg() << endl;
        throw;
    }
}

int main(int argc, char* argv[]) {
    // Check command-line arguments
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <X_file.h5> <W_file.h5>" << endl;
        cerr << "Example: " << argv[0] << " d5.h5 w5.h5" << endl;
        return 1;
    }
    string x_filename = argv[1];
    string w_filename = argv[2];
    
    cout << "=== Test Perm2D SpMM: List All Mismatches ===" << endl;
    cout << "Test workflow:" << endl;
    cout << "  1. Load " << x_filename << " and " << w_filename << endl;
    cout << "  2. Permute X row, X col, and W row" << endl;
    cout << "  3. Perform Y = X * W on permuted matrices" << endl;
    cout << "  4. Unpermute row of Y" << endl;
    cout << "  5. Compare with baseline Y and list ALL mismatches" << endl;
    cout << "  6. Save computed Y to file" << endl;
    cout << endl;
    
    try {
        
        string x_path = "../dataset/X/" + x_filename;
        string w_path = "../dataset/W/" + w_filename;
        
        cout << "Loading matrices..." << endl;
        cout << "  X from: " << x_path << endl;
        CSR X_original = load_X_h5_as_csr(x_path, "");
        cout << "  X: Rows=" << X_original.nrows << ", Cols=" << X_original.ncols 
             << ", nnz=" << X_original.nnz << endl;
        
        cout << "  W from: " << w_path << endl;
        int W_rows, W_cols;
        vector<float> W_original = load_W_h5(w_path, W_rows, W_cols, "");
        cout << "  W: Rows=" << W_rows << ", Cols=" << W_cols << endl;
        
        // Verify dimensions
        if (X_original.ncols != W_rows) {
            cerr << "  ✗ Dimension mismatch: X.ncols (" << X_original.ncols 
                 << ") != W.rows (" << W_rows << ")" << endl;
            return 1;
        }
        
        // ============================================================
        // Step 1: Permute row X
        // ============================================================
        cout << "\nStep 1: Permute row X" << endl;
        vector<size_t> nnz_per_row = compute_nnz_per_row(X_original);
        vector<int> row_new2old = create_row_new2old(nnz_per_row, true);
        CSR X_row_permuted = permute_csr_rows(X_original, row_new2old);
        cout << "  Permuted X rows: Rows=" << X_row_permuted.nrows 
             << ", Cols=" << X_row_permuted.ncols << ", nnz=" << X_row_permuted.nnz << endl;
        
        // ============================================================
        // Step 2: Permute col X and row W
        // ============================================================
        cout << "\nStep 2: Permute col X and row W" << endl;
        vector<size_t> nnz_per_col = compute_nnz_per_col(X_row_permuted);
        vector<int> col_new2old = create_col_new2old(nnz_per_col, true);
        
        if (static_cast<int>(col_new2old.size()) != W_rows) {
            cerr << "  ✗ Column permutation size (" << col_new2old.size() 
                 << ") != W.rows (" << W_rows << ")" << endl;
            return 1;
        }
        
        // Permute columns of X
        CSR X_row_col_permuted = permute_csr_cols(X_row_permuted, col_new2old);
        cout << "  Permuted X columns: Rows=" << X_row_col_permuted.nrows 
             << ", Cols=" << X_row_col_permuted.ncols << ", nnz=" << X_row_col_permuted.nnz << endl;
        
        // Permute rows of W
        vector<float> W_row_permuted = permute_weight_rows(W_original, W_rows, W_cols, col_new2old);
        cout << "  Permuted W rows: Rows=" << W_rows << ", Cols=" << W_cols << endl;
        
        // ============================================================
        // Step 3: Perform Y = X * W on permuted matrices
        // ============================================================
        cout << "\nStep 3: Perform Y = X * W on permuted matrices" << endl;
        vector<float> Y_permuted = spmm_baseline(X_row_col_permuted, W_row_permuted, W_rows, W_cols);
        int Y_rows = X_row_col_permuted.nrows;
        int Y_cols = W_cols;
        cout << "  Y_permuted: Rows=" << Y_rows << ", Cols=" << Y_cols << endl;
        
        // ============================================================
        // Step 4: Unpermute row of Y
        // ============================================================
        cout << "\nStep 4: Unpermute row of Y" << endl;
        vector<float> Y_final = unpermute_rows(Y_permuted, Y_rows, Y_cols, row_new2old);
        cout << "  Y_final: Rows=" << Y_rows << ", Cols=" << Y_cols << endl;
        
        // ============================================================
        // Step 5: Compare with baseline Y and list ALL mismatches
        // ============================================================
        cout << "\nStep 5: Compare Y_final with baseline Y and list ALL mismatches" << endl;
        string postfix = extract_postfix(x_filename);
        string y_baseline_path = "../dataset/Y/y" + postfix + "_baseline.h5";
        cout << "  Loading baseline Y from: " << y_baseline_path << endl;
        
        int y_baseline_rows, y_baseline_cols;
        vector<float> Y_baseline = load_Y_h5(y_baseline_path, y_baseline_rows, y_baseline_cols);
        cout << "  Y_baseline: Rows=" << y_baseline_rows << ", Cols=" << y_baseline_cols << endl;
        
        // Verify dimensions
        if (Y_rows != y_baseline_rows || Y_cols != y_baseline_cols) {
            cerr << "  ✗ Dimension mismatch: Y_final (" << Y_rows << "x" << Y_cols 
                 << ") vs Y_baseline (" << y_baseline_rows << "x" << y_baseline_cols << ")" << endl;
            return 1;
        }
        
        // Find and list ALL mismatches
        vector<pair<size_t, pair<float, float>>> mismatches;
        for (size_t i = 0; i < Y_final.size(); i++) {
            double abs_error = fabs(Y_final[i] - Y_baseline[i]);
            if (!approx_equal(Y_final[i], Y_baseline[i])) {
                mismatches.push_back({i, {Y_baseline[i], Y_final[i]}});
            }
        }

        
        cout << "\n" << string(80, '=') << endl;
        cout << "MISMATCH REPORT" << endl;
        cout << string(80, '=') << endl;
        cout << "Total elements: " << Y_final.size() << endl;
        cout << "Mismatches found: " << mismatches.size() << endl;
        cout << "Absolute tolerance (ABS_TOL): " << ABS_TOL << endl;
        cout << "Relative tolerance (REL_TOL): " << REL_TOL << endl;
        cout << "\nFormat: [row, col] expected observed" << endl;
        cout << string(80, '-') << endl;
        
        if (mismatches.empty()) {
            cout << "✓ No mismatches found! All elements match within tolerance." << endl;
        } else {
            cout << fixed << setprecision(10);
            for (const auto& mismatch : mismatches) {
                size_t idx = mismatch.first;
                float expected = mismatch.second.first;
                float observed = mismatch.second.second;
                int row = idx / Y_cols;
                int col = idx % Y_cols;
                double abs_error = fabs(expected - observed);
                
                cout << "[" << setw(5) << row << ", " << setw(5) << col << "] " 
                     << setw(15) << expected << " " << setw(15) << observed 
                     << " (error: " << abs_error << ")" << endl;
            }
        }
        
        cout << string(80, '=') << endl;
        
        // ============================================================
        // Step 6: Save computed Y to file
        // ============================================================
        cout << "\nStep 6: Save computed Y to file" << endl;
        string y_output_path = "../dataset/Y/y" + postfix + "_permspmm.h5";
        save_Y_h5(Y_final, Y_rows, Y_cols, y_output_path);
        cout << "  Saved Y_final to: " << y_output_path << endl;
        
        if (mismatches.empty()) {
            cout << "\n✓ Test PASSED: All elements match!" << endl;
            return 0;
        } else {
            cout << "\n✗ Test FAILED: " << mismatches.size() << " mismatches found!" << endl;
            return 1;
        }
        
    } catch (const exception& e) {
        cerr << "  ✗ Error: " << e.what() << endl;
        return 1;
    }
}

