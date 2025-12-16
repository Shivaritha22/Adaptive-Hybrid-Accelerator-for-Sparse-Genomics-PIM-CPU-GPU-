#include "../include/tile_spmm.hpp"
#include "../include/spmm.hpp"
#include "../include/logger.hpp"
#include "../include/dense_spmm_cuda.hpp"
#include "../config/hw_config.h"
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <omp.h>
#include <cstring>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

CSR extract_tile_csr(const CSR& X, const Tile& tile) {
    CSR X_tile;
    X_tile.nrows = tile.row_end - tile.row_start;
    X_tile.ncols = tile.col_end - tile.col_start;
    X_tile.nnz = 0;
    
    // First pass: count nnz per row in the tile
    X_tile.indptr.assign(X_tile.nrows + 1, 0);
    
    for (int i = tile.row_start; i < tile.row_end; i++) {
        int tile_row = i - tile.row_start;
        int row_start = X.indptr[i];
        int row_end = X.indptr[i + 1];
        
        int row_nnz = 0;
        for (int idx = row_start; idx < row_end; idx++) {
            int col = X.indices[idx];
            if (col >= tile.col_start && col < tile.col_end) {
                row_nnz++;
            }
        }
        X_tile.indptr[tile_row + 1] = X_tile.indptr[tile_row] + row_nnz;
        X_tile.nnz += row_nnz;
    }
    
    // Second pass: copy data with remapped column indices
    X_tile.indices.resize(X_tile.nnz);
    X_tile.data.resize(X_tile.nnz);
    
    vector<int> write_ptr = X_tile.indptr;
    for (int i = tile.row_start; i < tile.row_end; i++) {
        int tile_row = i - tile.row_start;
        int row_start = X.indptr[i];
        int row_end = X.indptr[i + 1];
        
        for (int idx = row_start; idx < row_end; idx++) {
            int col = X.indices[idx];
            if (col >= tile.col_start && col < tile.col_end) {
                int dest = write_ptr[tile_row]++;
                X_tile.indices[dest] = col - tile.col_start;  // Remap to 0-based
                X_tile.data[dest] = X.data[idx];
            }
        }
    }
    
    return X_tile;
}

vector<float> extract_tile_W(const vector<float>& W, int W_rows, int W_cols, const Tile& tile) {
    int W_tile_rows = tile.col_end - tile.col_start;
    vector<float> W_tile(W_tile_rows * W_cols, 0.0f);
    
    for (int i = 0; i < W_tile_rows; i++) {
        int orig_row = tile.col_start + i;
        if (orig_row >= 0 && orig_row < W_rows) {
            for (int j = 0; j < W_cols; j++) {
                W_tile[i * W_cols + j] = W[orig_row * W_cols + j];
            }
        }
    }
    
    return W_tile;
}

vector<float> materialize_csr_to_dense(const CSR& X_tile) {
    int M = X_tile.nrows;
    int K = X_tile.ncols;
    vector<float> X_dense(M * K, 0.0f);
    
    // Fill dense matrix from CSR
    for (int i = 0; i < M; i++) {
        int row_start = X_tile.indptr[i];
        int row_end = X_tile.indptr[i + 1];
        
        for (int idx = row_start; idx < row_end; idx++) {
            int k = X_tile.indices[idx];
            float val = X_tile.data[idx];
            X_dense[i * K + k] = val;
        }
    }
    
    return X_dense;
}

vector<float> permute_dense_rows(const vector<float>& X_dense, int M, int K,
                                const vector<int>& row_new2old) {
    vector<float> X_permuted(M * K, 0.0f);
    
    for (int new_row = 0; new_row < M; new_row++) {
        int old_row = row_new2old[new_row];
        for (int k = 0; k < K; k++) {
            X_permuted[new_row * K + k] = X_dense[old_row * K + k];
        }
    }
    
    return X_permuted;
}

vector<float> permute_dense_cols(const vector<float>& X_dense, int M, int K,
                                const vector<int>& col_new2old) {
    vector<float> X_permuted(M * K, 0.0f);
    
    for (int i = 0; i < M; i++) {
        for (int new_col = 0; new_col < K; new_col++) {
            int old_col = col_new2old[new_col];
            X_permuted[i * K + new_col] = X_dense[i * K + old_col];
        }
    }
    
    return X_permuted;
}

vector<float> dense_spmm_cpu_tile(const float* X_dense, const float* W_dense,
                                  int M, int K, int N) {
    vector<float> Y_dense(M * N, 0.0f);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += X_dense[i * K + k] * W_dense[k * N + j];
            }
            Y_dense[i * N + j] = sum;
        }
    }
    
    return Y_dense;
}

vector<float> dense_perm_spmm_tile(const CSR& X_tile, const vector<float>& W_tile, 
                                    int W_tile_rows, int W_cols) {
    int M_tile = X_tile.nrows;
    int K_tile = X_tile.ncols;
    int N = W_cols;
    
    // Step (a): Convert CSR tile to dense M×K buffer
    vector<float> X_dense = materialize_csr_to_dense(X_tile);
    
    // Step (b): Apply row/column permutation
    // Step 1: Permute tile rows (based on nnz per row from CSR)
    vector<size_t> nnz_per_row = compute_nnz_per_row(X_tile);
    vector<int> row_new2old = create_row_new2old(nnz_per_row, true);
    vector<float> X_dense_row_permuted = permute_dense_rows(X_dense, M_tile, K_tile, row_new2old);
    
    // Step 2: Permute tile cols and W_tile rows
    // We need to compute nnz per col from the permuted dense matrix
    // For efficiency, we can create a temporary CSR from the permuted dense to compute nnz_per_col
    // Or we can compute it directly from the dense matrix
    vector<size_t> nnz_per_col(K_tile, 0);
    for (int i = 0; i < M_tile; i++) {
        for (int k = 0; k < K_tile; k++) {
            if (X_dense_row_permuted[i * K_tile + k] != 0.0f) {
                nnz_per_col[k]++;
            }
        }
    }
    
    vector<int> col_new2old = create_col_new2old(nnz_per_col, true);
    
    if (static_cast<int>(col_new2old.size()) != W_tile_rows) {
        throw runtime_error("dense_perm_spmm_tile: column permutation size mismatch");
    }
    
    vector<float> X_dense_row_col_permuted = permute_dense_cols(X_dense_row_permuted, M_tile, K_tile, col_new2old);
    vector<float> W_tile_row_permuted = permute_weight_rows(W_tile, W_tile_rows, W_cols, col_new2old);
    
    // Step (c): Call either CUDA or CPU GEMM
    vector<float> Y_tile_permuted(M_tile * N, 0.0f);
    
    #ifdef USE_CUDA
    dense_spmm_cuda_tile(
        X_dense_row_col_permuted.data(),  // M_tile × K_tile
        W_tile_row_permuted.data(),       // K_tile × N
        Y_tile_permuted.data(),            // M_tile × N
        M_tile, K_tile, N);
    #else
    Y_tile_permuted = dense_spmm_cpu_tile(
        X_dense_row_col_permuted.data(),   // M_tile × K_tile
        W_tile_row_permuted.data(),       // K_tile × N
        M_tile, K_tile, N);
    #endif
    
    // Step (d): Unpermute the Y rows and return
    vector<float> Y_tile = unpermute_rows(Y_tile_permuted, M_tile, N, row_new2old);
    
    return Y_tile;
}

vector<float> sparse_spmm_tile(const CSR& X_tile, const vector<float>& W_tile, 
                               int W_tile_rows, int W_cols) {
    // Direct SpMM without permutation
    return spmm_baseline(X_tile, W_tile, W_tile_rows, W_cols);
}

vector<float> process_tiles_with_predictor(const CSR& X_original, 
                                          const vector<float>& W_original,
                                          int W_rows, int W_cols,
                                          const vector<Tile>& tiles,
                                          const string& log_annotation) {
    // Log OpenMP thread information
    if (!log_annotation.empty()) {
        int max_threads = omp_get_max_threads();
        log_openmp_threads_tilepredpermspmm(log_annotation, max_threads);
        
        // Log CUDA device information 
        #ifdef USE_CUDA
        CudaDeviceInfo cuda_info = get_cuda_device_info();
        log_cuda_device_info_tilepredpermspmm(log_annotation, cuda_info);
        #endif
    }
    
    int Y_rows = X_original.nrows;
    int Y_cols = W_cols;
    vector<float> Y_final(Y_rows * Y_cols, 0.0f);
    
    // Start timing
    auto start_time = high_resolution_clock::now();
    
    // Accumulate metrics
    size_t total_nnz = 0;
    size_t total_flops = 0;
    size_t total_bytes = 0;
    #ifdef USE_CUDA
    size_t cuda_dense_tiles = 0;
    #endif
    size_t cpu_dense_tiles = 0;
    
    // Process each tile
    for (const auto& tile : tiles) {
        // Extract tile as standalone CSR
        CSR X_tile = extract_tile_csr(X_original, tile);
        
        // Extract corresponding W rows
        vector<float> W_tile = extract_tile_W(W_original, W_rows, W_cols, tile);
        int W_tile_rows = tile.col_end - tile.col_start;
        
        vector<float> Y_tile;
        
        // Route based on density threshold
        if (tile.density() >= hw_config::DENSE_TILE_THRESHOLD) {
            // Dense tile: use dense materialization + CUDA/CPU GEMM
            Y_tile = dense_perm_spmm_tile(X_tile, W_tile, W_tile_rows, W_cols);
            #ifdef USE_CUDA
            cuda_dense_tiles++;
            #else
            cpu_dense_tiles++;
            #endif
        } else {
            // Sparse tile: direct SpMM (CSR-based with OpenMP)
            Y_tile = sparse_spmm_tile(X_tile, W_tile, W_tile_rows, W_cols);
        }
        
        // Accumulate metrics
        total_nnz += X_tile.nnz;
        // FLOPS: 2 * nnz * W_cols (multiply-add per nonzero)
        total_flops += static_cast<size_t>(2) * X_tile.nnz * W_cols;
        // Bytes: X data + X indices + X indptr + W + Y (read + write)
        size_t bytes_X_data = X_tile.nnz * sizeof(float);
        size_t bytes_X_indices = X_tile.nnz * sizeof(int);
        size_t bytes_X_indptr = (X_tile.nrows + 1) * sizeof(int);
        size_t bytes_W = static_cast<size_t>(W_tile_rows) * W_cols * sizeof(float);
        size_t bytes_Y = static_cast<size_t>(X_tile.nrows) * W_cols * sizeof(float) * 2; // read + write
        total_bytes += bytes_X_data + bytes_X_indices + bytes_X_indptr + bytes_W + bytes_Y;
        
        // Accumulate results into final Y (map tile rows back to global rows)
        int tile_rows = tile.row_end - tile.row_start;
        for (int i = 0; i < tile_rows; i++) {
            int global_row = tile.row_start + i;
            for (int j = 0; j < Y_cols; j++) {
                Y_final[global_row * Y_cols + j] += Y_tile[i * Y_cols + j];
            }
        }
    }
    
    // End timing
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    double compute_time_ms = duration.count() / 1000.0;
    
    // Log CUDA usage statistics
    if (!log_annotation.empty()) {
        #ifdef USE_CUDA
        log_cuda_usage_stats_tilepredpermspmm(log_annotation, cuda_dense_tiles, cpu_dense_tiles);
        #else
        // Log CPU usage when CUDA is not available
        stringstream ss;
        ss << "CUDA dense tiles: 0" << endl;
        ss << "CPU dense tiles: " << cpu_dense_tiles << endl;
        log_to_file_tilepredpermspmm(log_annotation, ss.str());
        #endif
    }
    
    // Log metrics
    if (!log_annotation.empty()) {
        // Calculate matrix density
        double matrix_density = 0.0;
        if (X_original.nrows > 0 && X_original.ncols > 0) {
            matrix_density = static_cast<double>(X_original.nnz) / 
                           (static_cast<double>(X_original.nrows) * static_cast<double>(X_original.ncols));
        }
        
        // Log all metrics
        stringstream ss;
        ss << fixed << setprecision(6);
        ss << "matrix_density: " << matrix_density << endl;
        log_to_file_tilepredpermspmm(log_annotation, ss.str());
        
        // Log SpMM metrics (accumulated)
        string log_filename = log_file_path_tilepredpermspmm(log_annotation);
        fs::create_directories(fs::path(log_filename).parent_path());
        
        const string time_prefix = "spmm compute time: ";
        const string nnz_prefix = "spmm nnz: ";
        const string flops_prefix = "spmm flops: ";
        const string bytes_prefix = "spmm bytes: ";
        const string perf_prefix = "spmm performance:";
        const string omp_prefix = "OpenMP threads: ";
        
        vector<string> preserved_lines;
        double existing_time_ms = 0.0;
        size_t existing_nnz = 0;
        double existing_flops = 0.0;
        double existing_bytes = 0.0;
        
        ifstream in(log_filename);
        if (in.is_open()) {
            string line;
            while (getline(in, line)) {
                if (line.rfind(time_prefix, 0) == 0) {
                    string value = line.substr(time_prefix.size());
                    size_t pos = value.find("ms");
                    if (pos != string::npos) value = value.substr(0, pos);
                    try {
                        existing_time_ms = stod(value);
                    } catch (...) {}
                    continue;
                }
                if (line.rfind(nnz_prefix, 0) == 0) {
                    string value = line.substr(nnz_prefix.size());
                    try {
                        existing_nnz = static_cast<size_t>(stoull(value));
                    } catch (...) {}
                    continue;
                }
                if (line.rfind(flops_prefix, 0) == 0) {
                    string value = line.substr(flops_prefix.size());
                    try {
                        existing_flops = stod(value);
                    } catch (...) {}
                    continue;
                }
                if (line.rfind(bytes_prefix, 0) == 0) {
                    string value = line.substr(bytes_prefix.size());
                    try {
                        existing_bytes = stod(value);
                    } catch (...) {}
                    continue;
                }
                if (line.rfind(perf_prefix, 0) == 0) {
                    continue;
                }
                // Preserve OpenMP threads line and all other lines
                preserved_lines.push_back(line);
            }
            in.close();
        }
        
        double total_time_ms = existing_time_ms + compute_time_ms;
        size_t total_nnz_accum = existing_nnz + total_nnz;
        double total_flops_accum = existing_flops + static_cast<double>(total_flops);
        double total_bytes_accum = existing_bytes + static_cast<double>(total_bytes);
        
        ofstream out(log_filename, ios::trunc);
        if (out.is_open()) {
            for (const auto& preserved : preserved_lines) {
                out << preserved << endl;
            }
            out << fixed << setprecision(3);
            out << "spmm compute time: " << total_time_ms << "ms" << endl;
            out << "spmm nnz: " << total_nnz_accum << endl;
            out << setprecision(3);
            out << flops_prefix << total_flops_accum << endl;
            out << bytes_prefix << total_bytes_accum << endl;
            
            double total_time_s = total_time_ms / 1000.0;
            if (total_time_s > 0 && (total_flops_accum > 0 || total_bytes_accum > 0)) {
                double gflops = (total_flops_accum > 0) ? (total_flops_accum / 1e9) / total_time_s : 0.0;
                double gbps = (total_bytes_accum > 0) ? (total_bytes_accum / 1e9) / total_time_s : 0.0;
                out << setprecision(2) << "spmm performance: " << gflops << " GFLOP/s, " << gbps << " GB/s" << endl;
            }
        }
    }
    
    return Y_final;
}

