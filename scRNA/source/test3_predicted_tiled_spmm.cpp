#include "../include/disk_to_memory.hpp"
#include "../include/tiler.hpp"
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

void compare_results(const vector<float>& Y_predicted_tiled, const vector<float>& Y_baseline, 
                    int rows, int cols, const string& test_name) {
    cout << "\n" << string(60, '=') << endl;
    cout << "COMPARISON: " << test_name << endl;
    cout << string(60, '=') << endl;
    
    
    if (Y_predicted_tiled.size() != Y_baseline.size()) {
        cerr << "ERROR: Size mismatch!" << endl;
        cerr << "  Y_predicted_tiled size: " << Y_predicted_tiled.size() << endl;
        cerr << "  Y_baseline size: " << Y_baseline.size() << endl;
        cerr << "  Expected: " << (rows * cols) << " (rows=" << rows << ", cols=" << cols << ")" << endl;
        return;
    }
    
    if (Y_predicted_tiled.size() != static_cast<size_t>(rows * cols)) {
        cerr << "ERROR: Size doesn't match dimensions!" << endl;
        cerr << "  Actual size: " << Y_predicted_tiled.size() << endl;
        cerr << "  Expected: " << (rows * cols) << " (rows=" << rows << ", cols=" << cols << ")" << endl;
        return;
    }
    
    
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double sum_abs_error = 0.0;
    double sum_sq_error = 0.0;
    size_t max_abs_error_idx = 0;
    size_t max_rel_error_idx = 0;
    size_t num_nonzero_baseline = 0;
    
    for (size_t i = 0; i < Y_predicted_tiled.size(); i++) {
        double abs_error = fabs(Y_predicted_tiled[i] - Y_baseline[i]);
        double rel_error = 0.0;
        
        sum_abs_error += abs_error;
        sum_sq_error += abs_error * abs_error;
        
        if (fabs(Y_baseline[i]) > 1e-10) {
            rel_error = abs_error / fabs(Y_baseline[i]);
            num_nonzero_baseline++;
        } else if (abs_error > 1e-10) {
            rel_error = abs_error;  
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
    
    size_t total_elements = Y_predicted_tiled.size();
    double mean_abs_error = sum_abs_error / total_elements;
    double rms_error = sqrt(sum_sq_error / total_elements);
    
    cout << fixed << setprecision(10);
    cout << "\nMatrix Dimensions: " << rows << " x " << cols << endl;
    cout << "Total Elements: " << total_elements << endl;
    cout << "Non-zero Baseline Elements: " << num_nonzero_baseline << endl;
    
    cout << "\n--- Error Statistics ---" << endl;
    cout << "  Mean absolute error: " << mean_abs_error << endl;
    cout << "  RMS error: " << rms_error << endl;
    cout << "  Max absolute error: " << max_abs_error << " (at index " << max_abs_error_idx << ")" << endl;
    cout << "    Y_predicted_tiled[" << max_abs_error_idx << "] = " << Y_predicted_tiled[max_abs_error_idx] << endl;
    cout << "    Y_baseline[" << max_abs_error_idx << "] = " << Y_baseline[max_abs_error_idx] << endl;
    cout << "    Difference: " << (Y_predicted_tiled[max_abs_error_idx] - Y_baseline[max_abs_error_idx]) << endl;
    
    if (num_nonzero_baseline > 0) {
        cout << "  Max relative error: " << max_rel_error << " (at index " << max_rel_error_idx << ")" << endl;
        cout << "    Y_predicted_tiled[" << max_rel_error_idx << "] = " << Y_predicted_tiled[max_rel_error_idx] << endl;
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
        cout << "\n✓ Predictor + Tiling produces mathematically equivalent results!" << endl;
        cout << "  (Differences are within floating-point numerical precision)" << endl;
    } else {
        cout << "\n✗ Results do not match within tolerance!" << endl;
        cout << "  This indicates a potential issue with the predictor or tiling logic." << endl;
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
    string y_predicted_tiled_path = "../dataset/Y/y" + postfix + "_predicted_tiled.h5";
    
    
    string log_annotation = extract_postfix(x_filename);
    reset_log(log_annotation);
    
    cout << "=== Test 3: Predictor + Tiled SpMM (PIM OFF) ===" << endl;
    cout << "X file: " << x_path << endl;
    cout << "W file: " << w_path << endl;
    cout << "Baseline Y file: " << y_baseline_path << endl;
    cout << "Output Y file: " << y_predicted_tiled_path << endl;
    
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
        
        
        if (y_rows != X.nrows || y_cols != w_cols) {
            cerr << "ERROR: Baseline Y dimensions don't match expected!" << endl;
            cerr << "  Expected: " << X.nrows << " x " << w_cols << endl;
            cerr << "  Actual: " << y_rows << " x " << y_cols << endl;
            return 1;
        }
        
        // Create tiling config and apply 2D tiling with predictor
        cout << "\nApplying 2D tiling with density predictor..." << endl;
        TilingConfig cfg;
        cout << "Tiling config: " << cfg.tile_rows << " x " << cfg.tile_cols << endl;
        
        // Verify dimensions match before computation
        if (X.ncols != w_rows) {
            cerr << "ERROR: Dimension mismatch!" << endl;
            cerr << "  X.ncols: " << X.ncols << endl;
            cerr << "  w_rows: " << w_rows << endl;
            return 1;
        }
        
        // Run tiled SpMM with predictor (dense tiles → dense_spmm_tile, sparse tiles → traditional path)
        cout << "\nRunning tiled SpMM with density predictor..." << endl;
        auto start = chrono::high_resolution_clock::now();
        auto result = spmm_tiled(X, W, w_rows, w_cols, cfg, log_annotation);
        vector<float> Y_predicted_tiled = result.first;
        size_t num_tiles = result.second;
        auto end = chrono::high_resolution_clock::now();
        
        // Calculate compute time
        auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start).count();
        double duration_ms = duration_us / 1000.0;
        
        cout << "Y_predicted_tiled: " << X.nrows << " x " << w_cols << endl;
        cout << "Compute time: " << fixed << setprecision(3) << duration_ms << " ms" << endl;
        cout << "Number of tiles: " << num_tiles << endl;
        
        // Log compute time and nnz using annotation-based logging
        // (tile count, dense/sparse counts, and matrix density are already logged by spmm_tiled)
        long long flops = static_cast<long long>(X.nnz) * static_cast<long long>(w_cols) * 2;
        size_t bytes_X_data = X.nnz * sizeof(float);
        size_t bytes_X_indices = X.nnz * sizeof(int);
        size_t bytes_X_indptr = (X.nrows + 1) * sizeof(int);
        size_t bytes_W = static_cast<size_t>(w_rows) * static_cast<size_t>(w_cols) * sizeof(float);
        size_t bytes_Y = static_cast<size_t>(X.nrows) * static_cast<size_t>(w_cols) * sizeof(float) * 2;
        size_t total_bytes = bytes_X_data + bytes_X_indices + bytes_X_indptr + bytes_W + bytes_Y;
        log_spmm_metrics(log_annotation, duration_ms, X.nnz,
                       static_cast<double>(flops),
                       static_cast<double>(total_bytes));
        
        silent_save_Y(Y_predicted_tiled, X.nrows, w_cols, y_predicted_tiled_path);
        cout << "Saved result to: " << y_predicted_tiled_path << endl;
        
        // Compare Y_predicted_tiled against baseline Y
        // This verifies that predictor + tiling produces mathematically equivalent results
        compare_results(Y_predicted_tiled, Y_baseline, X.nrows, w_cols, 
                       "Predictor + Tiled SpMM vs Baseline (from run0.exe)");
        
        return 0;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}











