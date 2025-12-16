#include "../include/disk_to_memory.hpp"
#include "../include/tiler.hpp"
#include "../include/permutation.hpp"
#include "../include/spmm.hpp"
#include "../include/csr.hpp"
#include "../include/logger.hpp"
#include <H5Cpp.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <filesystem>

using namespace std;
using namespace H5;
namespace fs = std::filesystem;


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

void silent_save_Y(const vector<float>& Y, int rows, int cols, const string& path) {
    try {
        fs::create_directories(fs::path(path).parent_path());
        H5File file(path, H5F_ACC_TRUNC);
        hsize_t dims[2] = {static_cast<hsize_t>(rows), static_cast<hsize_t>(cols)};
        DataSpace dataspace(2, dims);
        DataSet dataset = file.createDataSet("Y", PredType::NATIVE_FLOAT, dataspace);
        dataset.write(Y.data(), PredType::NATIVE_FLOAT);
    } catch (...) {}
}


void compare_results(const vector<float>& Y_perm_tiled, const vector<float>& Y_baseline, 
                    int rows, int cols, const string& test_name) {
    cout << "\n" << string(60, '=') << endl;
    cout << "COMPARISON: " << test_name << endl;
    cout << string(60, '=') << endl;
    
    // Validate dimensions
    if (Y_perm_tiled.size() != Y_baseline.size()) {
        cerr << "ERROR: Size mismatch!" << endl;
        cerr << "  Y_perm_tiled size: " << Y_perm_tiled.size() << endl;
        cerr << "  Y_baseline size: " << Y_baseline.size() << endl;
        cerr << "  Expected: " << (rows * cols) << " (rows=" << rows << ", cols=" << cols << ")" << endl;
        return;
    }
    
    if (Y_perm_tiled.size() != static_cast<size_t>(rows * cols)) {
        cerr << "ERROR: Size doesn't match dimensions!" << endl;
        cerr << "  Actual size: " << Y_perm_tiled.size() << endl;
        cerr << "  Expected: " << (rows * cols) << " (rows=" << rows << ", cols=" << cols << ")" << endl;
        return;
    }
    
    // Compute error statistics
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double sum_abs_error = 0.0;
    double sum_sq_error = 0.0;
    size_t max_abs_error_idx = 0;
    size_t max_rel_error_idx = 0;
    size_t num_nonzero_baseline = 0;
    
    for (size_t i = 0; i < Y_perm_tiled.size(); i++) {
        double abs_error = fabs(Y_perm_tiled[i] - Y_baseline[i]);
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
    
    size_t total_elements = Y_perm_tiled.size();
    double mean_abs_error = sum_abs_error / total_elements;
    double rms_error = sqrt(sum_sq_error / total_elements);
    
    // Output detailed results
    cout << fixed << setprecision(10);
    cout << "\nMatrix Dimensions: " << rows << " x " << cols << endl;
    cout << "Total Elements: " << total_elements << endl;
    cout << "Non-zero Baseline Elements: " << num_nonzero_baseline << endl;
    
    cout << "\n--- Error Statistics ---" << endl;
    cout << "  Mean absolute error: " << mean_abs_error << endl;
    cout << "  RMS error: " << rms_error << endl;
    cout << "  Max absolute error: " << max_abs_error << " (at index " << max_abs_error_idx << ")" << endl;
    cout << "    Y_perm_tiled[" << max_abs_error_idx << "] = " << Y_perm_tiled[max_abs_error_idx] << endl;
    cout << "    Y_baseline[" << max_abs_error_idx << "] = " << Y_baseline[max_abs_error_idx] << endl;
    cout << "    Difference: " << (Y_perm_tiled[max_abs_error_idx] - Y_baseline[max_abs_error_idx]) << endl;
    
    if (num_nonzero_baseline > 0) {
        cout << "  Max relative error: " << max_rel_error << " (at index " << max_rel_error_idx << ")" << endl;
        cout << "    Y_perm_tiled[" << max_rel_error_idx << "] = " << Y_perm_tiled[max_rel_error_idx] << endl;
        cout << "    Y_baseline[" << max_rel_error_idx << "] = " << Y_baseline[max_rel_error_idx] << endl;
        cout << "    Relative difference: " << (max_rel_error * 100.0) << "%" << endl;
    }
    
    // Check if errors are within floating-point tolerance
    const double FLOAT_TOL = 1e-5;
    bool passed = (max_abs_error < FLOAT_TOL) && (max_rel_error < FLOAT_TOL);
    
    cout << "\n--- Validation Result ---" << endl;
    cout << "  Tolerance: max_abs_error < " << FLOAT_TOL << ", max_rel_error < " << FLOAT_TOL << endl;
    cout << "  Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << endl;
    
    if (passed) {
        cout << "\n✓ Permutation + Tiling produces mathematically equivalent results!" << endl;
        cout << "  (Differences are within floating-point numerical precision)" << endl;
    } else {
        cout << "\n✗ Results do not match within tolerance!" << endl;
        cout << "  This indicates a potential issue with permutation or tiling logic." << endl;
    }
    
    cout << string(60, '=') << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <X_file.h5> <W_file.h5>" << endl;
        cerr << "Example: " << argv[0] << " d0.h5 w0.h5" << endl;
        return 1;
    }
    
    string x_filename = argv[1];  // e.g., d0.h5
    string w_filename = argv[2];  // e.g., w0.h5
    
    
    string x_path = "../dataset/X/" + x_filename;
    string w_path = "../dataset/W/" + w_filename;
    
    string postfix = extract_postfix(x_filename);
    string y_baseline_path = "../dataset/Y/y" + postfix + ".h5";
    string y_perm_tiled_path = "../dataset/Y/y" + postfix + "_perm_tiled.h5";
    
    
    string log_annotation = extract_postfix(x_filename);
    reset_log(log_annotation);
    
    cout << "=== Test 2: Permutation + Tiled SpMM (PIM OFF) ===" << endl;
    cout << "X file: " << x_path << endl;
    cout << "W file: " << w_path << endl;
    cout << "Baseline Y file: " << y_baseline_path << endl;
    cout << "Output Y file: " << y_perm_tiled_path << endl;
    
    try {
        
        cout << "\nLoading matrices..." << endl;
        CSR X = load_X_h5_as_csr(x_path, log_annotation);
        int w_rows, w_cols;
        vector<float> W = load_W_h5(w_path, w_rows, w_cols, log_annotation);
        
        cout << "X: " << X.nrows << " x " << X.ncols << ", nnz: " << X.nnz << endl;
        cout << "W: " << w_rows << " x " << w_cols << endl;
        

        cout << "\nLoading baseline Y (from run0.exe output)..." << endl;
        if (!fs::exists(y_baseline_path)) {
            cerr << "ERROR: Baseline Y file not found: " << y_baseline_path << endl;
            cerr << "Please run run0.exe first to generate the baseline Y file." << endl;
            cerr << "Example: .\\..\\build\\run0.exe " << x_filename << " " << w_filename << endl;
            return 1;
        }
        
        int y_rows, y_cols;
        vector<float> Y_baseline = load_Y_h5(y_baseline_path, y_rows, y_cols);
        cout << "Y_baseline: " << y_rows << " x " << y_cols << " (loaded from " << y_baseline_path << ")" << endl;
        
        // Validate dimensions match
        if (y_rows != X.nrows || y_cols != w_cols) {
            cerr << "ERROR: Baseline Y dimensions don't match expected!" << endl;
            cerr << "  Expected: " << X.nrows << " x " << w_cols << endl;
            cerr << "  Actual: " << y_rows << " x " << y_cols << endl;
            return 1;
        }
        
        // Step 1: Compute nnz statistics (row-only permutation)
        cout << "\nComputing nnz statistics..." << endl;
        vector<size_t> nnz_per_row = compute_nnz_per_row(X);
        
        // Step 2: Create row permutation mapping (new → old)
        cout << "Creating row permutation (new → old)..." << endl;
        vector<int> row_new2old = create_row_new2old(nnz_per_row, true);  // descending nnz
        cout << "  ✓ Row permutation mapping created" << endl;
        
        // Step 3: Permute X rows only: X → X_perm
        cout << "\nPermuting X rows..." << endl;
        CSR X_perm = permute_csr_rows(X, row_new2old);
        cout << "  ✓ X permuted to X_perm (row permutation only)" << endl;
        
        // Step 4: W is NOT permuted (row permutation on X only)
        // SpMM will use X_perm and W directly
        cout << "  ✓ W kept unchanged (no permutation needed)" << endl;
        
        // Step 5: Create tiling config and apply 2D tiling on X_perm
        cout << "\nApplying 2D tiling on X_perm..." << endl;
        TilingConfig cfg;
        cout << "Tiling config: " << cfg.tile_rows << " x " << cfg.tile_cols << endl;
        
        // Step 6: Run tiled SpMM on X_perm, W → Y'
        // Verify dimensions match before computation
        if (X_perm.ncols != w_rows) {
            cerr << "ERROR: Dimension mismatch!" << endl;
            cerr << "  X_perm.ncols: " << X_perm.ncols << endl;
            cerr << "  w_rows: " << w_rows << endl;
            return 1;
        }
        
        // For debugging: compute Y' using baseline SpMM to verify permutation is correct
        // This helps isolate whether the issue is in tiling or permutation
        vector<float> Y_prime_baseline = spmm_baseline(X_perm, W, X_perm.ncols, w_cols);
        
        cout << "\nRunning tiled SpMM on X_perm, W..." << endl;
        auto start = chrono::high_resolution_clock::now();
        auto result = spmm_tiled(X_perm, W, X_perm.ncols, w_cols, cfg, log_annotation);
        vector<float> Y_prime = result.first;
        size_t num_tiles = result.second;
        auto end = chrono::high_resolution_clock::now();
        
        // Verify tiled result matches baseline result on permuted matrices
        bool tiled_matches_baseline = true;
        double max_tiled_error = 0.0;
        size_t max_tiled_error_idx = 0;
        for (size_t i = 0; i < Y_prime.size(); i++) {
            double error = fabs(Y_prime[i] - Y_prime_baseline[i]);
            if (error > 1e-5) {
                tiled_matches_baseline = false;
                if (error > max_tiled_error) {
                    max_tiled_error = error;
                    max_tiled_error_idx = i;
                }
            }
        }
        if (!tiled_matches_baseline) {
            cerr << "WARNING: Tiled SpMM on permuted matrices doesn't match baseline SpMM!" << endl;
            cerr << "  Max error: " << max_tiled_error << " (at index " << max_tiled_error_idx << ")" << endl;
            cerr << "  Y_prime[tiled][" << max_tiled_error_idx << "] = " << Y_prime[max_tiled_error_idx] << endl;
            cerr << "  Y_prime_baseline[" << max_tiled_error_idx << "] = " << Y_prime_baseline[max_tiled_error_idx] << endl;
            cerr << "  This indicates an issue with the tiled SpMM implementation." << endl;
        } else {
            cout << "  ✓ Tiled SpMM on permuted matrices matches baseline SpMM (within tolerance)" << endl;
        }
        
        // Calculate compute time
        auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start).count();
        double duration_ms = duration_us / 1000.0;
        
        cout << "Y': " << X_perm.nrows << " x " << w_cols << endl;
        cout << "Compute time: " << fixed << setprecision(3) << duration_ms << " ms" << endl;
        cout << "Number of tiles: " << num_tiles << endl;
        
        // Log compute time and nnz using annotation-based logging
        // (tile count is already logged by make_2d_tiles)
        long long flops = static_cast<long long>(X_perm.nnz) * static_cast<long long>(w_cols) * 2;
        size_t bytes_X_data = X_perm.nnz * sizeof(float);
        size_t bytes_X_indices = X_perm.nnz * sizeof(int);
        size_t bytes_X_indptr = (X_perm.nrows + 1) * sizeof(int);
        size_t bytes_W = static_cast<size_t>(w_rows) * static_cast<size_t>(w_cols) * sizeof(float);
        size_t bytes_Y = static_cast<size_t>(X_perm.nrows) * static_cast<size_t>(w_cols) * sizeof(float) * 2;
        size_t total_bytes = bytes_X_data + bytes_X_indices + bytes_X_indptr + bytes_W + bytes_Y;
        log_spmm_metrics(log_annotation, duration_ms, X_perm.nnz,
                       static_cast<double>(flops),
                       static_cast<double>(total_bytes));
        
        // Step 7: Unpermute Y' rows to get Y_perm_tiled
        cout << "\nUnpermuting result rows..." << endl;
        cout << "  Y' dimensions: " << X_perm.nrows << " x " << w_cols << endl;
        cout << "  row_new2old size: " << row_new2old.size() << endl;
        cout << "  Expected Y dimensions: " << X.nrows << " x " << w_cols << endl;
        vector<float> Y_perm_tiled = unpermute_rows(Y_prime, X.nrows, w_cols, row_new2old);
        cout << "  ✓ Y' unpermuted to Y_perm_tiled" << endl;
        
        // Debug: Check if unpermutation is correct by comparing with baseline on permuted matrices
        // If we unpermute Y_prime_baseline, we should get the same as Y_baseline
        vector<float> Y_recovered_from_baseline = unpermute_rows(Y_prime_baseline, X.nrows, w_cols, row_new2old);
        bool unperm_correct = true;
        double max_unperm_error = 0.0;
        size_t max_unperm_error_idx = 0;
        for (size_t i = 0; i < Y_baseline.size(); i++) {
            double error = fabs(Y_recovered_from_baseline[i] - Y_baseline[i]);
            if (error > 1e-5) {
                unperm_correct = false;
                if (error > max_unperm_error) {
                    max_unperm_error = error;
                    max_unperm_error_idx = i;
                }
            }
        }
        if (!unperm_correct) {
            cerr << "WARNING: Unpermutation of Y_prime_baseline doesn't match Y_baseline!" << endl;
            cerr << "  Max error: " << max_unperm_error << " (at index " << max_unperm_error_idx << ")" << endl;
            int row_idx = max_unperm_error_idx / w_cols;
            int col_idx = max_unperm_error_idx % w_cols;
            cerr << "  Row: " << row_idx << ", Col: " << col_idx << endl;
            cerr << "  Y_baseline[" << max_unperm_error_idx << "] = " << Y_baseline[max_unperm_error_idx] << endl;
            cerr << "  Y_recovered[" << max_unperm_error_idx << "] = " << Y_recovered_from_baseline[max_unperm_error_idx] << endl;
            cerr << "  This indicates an issue with the unpermutation logic." << endl;
        } else {
            cout << "  ✓ Unpermutation logic verified (Y_prime_baseline unpermuted matches Y_baseline)" << endl;
        }
        
        // Save result to y{postfix}_perm_tiled.h5
        silent_save_Y(Y_perm_tiled, X.nrows, w_cols, y_perm_tiled_path);
        cout << "Saved result to: " << y_perm_tiled_path << endl;
        
        // Step 8: Compare Y_perm_tiled against baseline Y (from run0.exe)
        // This verifies that permutation + tiling + unpermutation produces
        // mathematically equivalent results to the baseline computation
        compare_results(Y_perm_tiled, Y_baseline, X.nrows, w_cols, 
                       "Permuted + Tiled SpMM vs Baseline (from run0.exe)");
        
        return 0;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
