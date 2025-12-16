#include "../include/disk_to_memory.hpp"
#include "../include/spmm.hpp"
#include "../include/csr.hpp"
#include "../include/logger.hpp"
#include <H5Cpp.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace std;
using namespace H5;
namespace fs = std::filesystem;

// Function to extract postfix from filename (e.g., "d0.h5" -> "0")
string extract_postfix(const string& filename) {
    
    string name = filename;
    size_t dot_pos = name.find_last_of('.');
    if (dot_pos != string::npos) {
        name = name.substr(0, dot_pos);
    }
    
    if (name.length() > 1) {
        return name.substr(1); // Skip first character (e.g., 'd' in "d0")
    }
    return "0";
}

// Function to generate log filename based on input postfix
string generate_log_filename(const string& x_filename) {
    string postfix = extract_postfix(x_filename);
    return "../logs/log" + postfix + ".txt";
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

/**
 * Baseline run wrapper - times spmm_baseline(), computes performance metrics, and logs.
 * Takes already-loaded matrices. Can be reused by PIM/hybrid implementations for comparison.
 * 
 * @param X CSR sparse matrix (already loaded)
 * @param W Dense matrix as vector<float> (already loaded)
 * @param W_rows Number of rows in W
 * @param W_cols Number of columns in W
 * @param log_annotation log file annotation
 * @return Result vector Y = X * W
 */
vector<float> baseline_run(const CSR& X, const vector<float>& W, int W_rows, int W_cols, const string& log_annotation = "") {
    // Start timing
    auto start = chrono::high_resolution_clock::now();
    
    // Perform computation
    vector<float> Y = spmm_baseline(X, W, W_rows, W_cols, log_annotation);
    
    // End timing
    auto end = chrono::high_resolution_clock::now();
    auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double duration_ms = duration_us / 1000.0;
    
    // Compute FLOPs: For SpMM Y = X * W, each non-zero in X contributes W_cols multiply-adds
    // So total FLOPs = nnz_X * W_cols * 2 (multiply + add)
    long long flops = static_cast<long long>(X.nnz) * static_cast<long long>(W_cols) * 2;
    
    // Compute memory bandwidth:
    // - Read X: nnz * sizeof(float) for data + nnz * sizeof(int) for indices + (nrows+1) * sizeof(int) for indptr
    // - Read W: W_rows * W_cols * sizeof(float)
    // - Read/Write Y: X.nrows * W_cols * sizeof(float) * 2 (read and write)
    size_t bytes_X_data = X.nnz * sizeof(float);
    size_t bytes_X_indices = X.nnz * sizeof(int);
    size_t bytes_X_indptr = (X.nrows + 1) * sizeof(int);
    size_t bytes_W = static_cast<size_t>(W_rows) * static_cast<size_t>(W_cols) * sizeof(float);
    size_t bytes_Y = static_cast<size_t>(X.nrows) * static_cast<size_t>(W_cols) * sizeof(float) * 2; // read + write
    size_t total_bytes = bytes_X_data + bytes_X_indices + bytes_X_indptr + bytes_W + bytes_Y;
    
    if (!log_annotation.empty()) {
        log_spmm_metrics(log_annotation, duration_ms, X.nnz, static_cast<double>(flops), static_cast<double>(total_bytes));
    }
    
    return Y;
}

/*
  Baseline run function - loads matrices from disk, runs baseline computation, saves results.
  This is the full pipeline function that handles I/O.
 
  @param x_path Path to input X matrix (sparse, HDF5 format)
  @param w_path Path to input W matrix (dense, HDF5 format)
  @param y_path Path to output Y matrix (will be overwritten)
  @param log_annotation Optional log file annotation (e.g., "0" for log0.txt).
  @return true on success, false on failure
 */
bool baseline_run_from_disk(const string& x_path, const string& w_path, const string& y_path, const string& log_annotation = "") {
    try {
        // Load X matrix (logs: rows_X, cols_X, nnz_X and X load time)
        CSR X = load_X_h5_as_csr(x_path, log_annotation);
        
        // Load W matrix (logs: rows_W, cols_W and W load time)
        int w_rows, w_cols;
        vector<float> W = load_W_h5(w_path, w_rows, w_cols, log_annotation);
        
    // Perform SpMM computation
    vector<float> Y = baseline_run(X, W, w_rows, w_cols, log_annotation);

        silent_save_Y(Y, X.nrows, w_cols, y_path);
        
        return true;
    } catch (const exception& e) {
        cerr << "Error in baseline_run_from_disk: " << e.what() << endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) return 1;

    string x_in = argv[1];
    string w_in = argv[2];

    string log_annotation = extract_postfix(x_in);
    
    reset_log(log_annotation);
    
    std::cout.setstate(std::ios_base::failbit);

    string x_path = "../dataset/X/" + x_in;
    string w_path = "../dataset/W/" + w_in;
    
    string postfix = extract_postfix(x_in);
    string y_path = "../dataset/Y/y" + postfix + ".h5";

    bool success = baseline_run_from_disk(x_path, w_path, y_path, log_annotation);

    std::cout.clear();
    
    if (success) {
        cout << "spmm done" << endl;
        return 0;
    } else {
        cerr << "spmm failed" << endl;
        return 1;
    }
}