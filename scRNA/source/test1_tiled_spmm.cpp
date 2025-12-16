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
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <fstream>
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


void compare_results(const vector<float>& Y_tiled, const vector<float>& Y_baseline, 
                    int rows, int cols, const string& test_name) {
    if (Y_tiled.size() != Y_baseline.size()) {
        cerr << "Error: Size mismatch! Y_tiled=" << Y_tiled.size() 
             << ", Y_baseline=" << Y_baseline.size() << endl;
        return;
    }
    
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    size_t max_abs_error_idx = 0;
    size_t max_rel_error_idx = 0;
    
    for (size_t i = 0; i < Y_tiled.size(); i++) {
        double abs_error = fabs(Y_tiled[i] - Y_baseline[i]);
        double rel_error = 0.0;
        
        if (fabs(Y_baseline[i]) > 1e-10) {
            rel_error = abs_error / fabs(Y_baseline[i]);
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
    
    // Output results
    cout << fixed << setprecision(10);
    cout << "\n=== " << test_name << " ===" << endl;
    cout << "Matrix dimensions: " << rows << " x " << cols << endl;
    cout << "Total elements: " << Y_tiled.size() << endl;
    cout << "\nError Metrics:" << endl;
    cout << "  Max absolute error: " << max_abs_error << " (at index " << max_abs_error_idx << ")" << endl;
    cout << "    Y_tiled[" << max_abs_error_idx << "] = " << Y_tiled[max_abs_error_idx] << endl;
    cout << "    Y_baseline[" << max_abs_error_idx << "] = " << Y_baseline[max_abs_error_idx] << endl;
    cout << "  Max relative error: " << max_rel_error << " (at index " << max_rel_error_idx << ")" << endl;
    cout << "    Y_tiled[" << max_rel_error_idx << "] = " << Y_tiled[max_rel_error_idx] << endl;
    cout << "    Y_baseline[" << max_rel_error_idx << "] = " << Y_baseline[max_rel_error_idx] << endl;
    
    // Check if errors are within floating-point tolerance
    const double FLOAT_TOL = 1e-5;
    bool passed = (max_abs_error < FLOAT_TOL) && (max_rel_error < FLOAT_TOL);
    
    cout << "\nResult: " << (passed ? "PASSED" : "FAILED") << endl;
    cout << "  (Expected: max_abs_error < " << FLOAT_TOL << ", max_rel_error < " << FLOAT_TOL << ")" << endl;
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
    string y_tile_path = "../dataset/Y/y" + postfix + "_tile.h5";
    
    
    string log_annotation = extract_postfix(x_filename);
    reset_log(log_annotation);
    
    cout << "=== Test 1: Tiled SpMM (PIM OFF, Permutation OFF) ===" << endl;
    cout << "X file: " << x_path << endl;
    cout << "W file: " << w_path << endl;
    cout << "Baseline Y file: " << y_baseline_path << endl;
    cout << "Output Y file: " << y_tile_path << endl;
    
    try {
        
        cout << "\nLoading matrices..." << endl;
        CSR X = load_X_h5_as_csr(x_path, log_annotation);
        int w_rows, w_cols;
        vector<float> W = load_W_h5(w_path, w_rows, w_cols, log_annotation);
        
        cout << "X: " << X.nrows << " x " << X.ncols << ", nnz: " << X.nnz << endl;
        cout << "W: " << w_rows << " x " << w_cols << endl;
        
        
        cout << "\nLoading baseline Y..." << endl;
        int y_rows, y_cols;
        vector<float> Y_baseline = load_Y_h5(y_baseline_path, y_rows, y_cols);
        cout << "Y_baseline: " << y_rows << " x " << y_cols << endl;
        
        
        TilingConfig cfg;
        cout << "\nTiling config: " << cfg.tile_rows << " x " << cfg.tile_cols << endl;
        
        
        cout << "\nRunning tiled SpMM..." << endl;
        auto start = chrono::high_resolution_clock::now();
        auto result = spmm_tiled(X, W, w_rows, w_cols, cfg, log_annotation);
        vector<float> Y_tiled = result.first;
        size_t num_tiles = result.second;
        auto end = chrono::high_resolution_clock::now();
        
        // Calculate compute time
        auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start).count();
        double duration_ms = duration_us / 1000.0;
        
        cout << "Y_tiled: " << X.nrows << " x " << w_cols << endl;
        cout << "Compute time: " << fixed << setprecision(3) << duration_ms << " ms" << endl;
        cout << "Number of tiles: " << num_tiles << endl;
        
        // Compute FLOPs and memory traffic for performance metrics
        long long flops = static_cast<long long>(X.nnz) * static_cast<long long>(w_cols) * 2;
        size_t bytes_X_data = X.nnz * sizeof(float);
        size_t bytes_X_indices = X.nnz * sizeof(int);
        size_t bytes_X_indptr = (X.nrows + 1) * sizeof(int);
        size_t bytes_W = static_cast<size_t>(w_rows) * static_cast<size_t>(w_cols) * sizeof(float);
        size_t bytes_Y = static_cast<size_t>(X.nrows) * static_cast<size_t>(w_cols) * sizeof(float) * 2;
        size_t total_bytes = bytes_X_data + bytes_X_indices + bytes_X_indptr + bytes_W + bytes_Y;
        
        // Log compute time, nnz, and accumulated performance metrics
        log_spmm_metrics(log_annotation, duration_ms, X.nnz, static_cast<double>(flops), static_cast<double>(total_bytes));
        
        
        silent_save_Y(Y_tiled, X.nrows, w_cols, y_tile_path);
        cout << "Saved result to: " << y_tile_path << endl;
        
        // Compare results
        compare_results(Y_tiled, Y_baseline, X.nrows, w_cols, "Tiled SpMM vs Baseline");
        
        return 0;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

