#pragma once
#include <string>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

inline string log_file_path(const string& annotation, const string& base_path = "../logs/") {
    if (annotation.empty()) {
        return base_path + "log_default.txt";
    }
    return base_path + "log" + annotation + ".txt";
}

inline void reset_log(const string& annotation, const string& base_path = "../logs/") {
    string log_filename = log_file_path(annotation, base_path);
    fs::create_directories(fs::path(log_filename).parent_path());
    ofstream log(log_filename, ios::trunc);
}

/**
 Logging utility for annotation-based logging.
 Functions can specify which log file to write to using an annotation (e.g., "0" -> "log0.txt").
 
 @param annotation Log file annotation (e.g., "0" for log0.txt, "1" for log1.txt, empty for default)
 @param message The log message to write
 @param base_path Base directory for logs (default: "../logs/")
 */
inline void log_to_file(const string& annotation, const string& message, const string& base_path = "../logs/") {
    // Generate log filename from annotation
    string log_filename = log_file_path(annotation, base_path);
    
    // Ensure directory exists
    fs::create_directories(fs::path(log_filename).parent_path());
    
    // Append to log file
    ofstream log(log_filename, ios::app);
    if (log.is_open()) {
        log << message;
        log.close();
    }
}

/**
 Helper function to format and log SpMM compute time and nnz
 */
inline void log_spmm_metrics(const string& annotation, double compute_time_ms, size_t nnz, 
                             double flops = 0.0, double bytes = 0.0) {
    string log_filename = log_file_path(annotation);
    fs::create_directories(fs::path(log_filename).parent_path());
    
    const string time_prefix = "spmm compute time: ";
    const string nnz_prefix = "spmm nnz: ";
    const string flops_prefix = "spmm flops: ";
    const string bytes_prefix = "spmm bytes: ";
    const string perf_prefix = "spmm performance:";
    
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
                // Skip old performance line; we'll rewrite it
                continue;
            }
            preserved_lines.push_back(line);
        }
        in.close();
    }
    
    double total_time_ms = existing_time_ms + compute_time_ms;
    size_t total_nnz = existing_nnz + nnz;
    double total_flops = existing_flops + flops;
    double total_bytes = existing_bytes + bytes;
    
    ofstream out(log_filename, ios::trunc);
    if (out.is_open()) {
        for (const auto& preserved : preserved_lines) {
            out << preserved << endl;
        }
        out << fixed << setprecision(3);
        out << "spmm compute time: " << total_time_ms << "ms" << endl;
        out << "spmm nnz: " << total_nnz << endl;
        out << setprecision(3);
        out << flops_prefix << total_flops << endl;
        out << bytes_prefix << total_bytes << endl;
        
        double total_time_s = total_time_ms / 1000.0;
        if (total_time_s > 0 && (total_flops > 0 || total_bytes > 0)) {
            double gflops = (total_flops > 0) ? (total_flops / 1e9) / total_time_s : 0.0;
            double gbps = (total_bytes > 0) ? (total_bytes / 1e9) / total_time_s : 0.0;
            out << setprecision(2) << "spmm performance: " << gflops << " GFLOP/s, " << gbps << " GB/s" << endl;
        }
    }
}

/**
 Helper function to log tiler metrics (number of tiles)
 */
inline void log_tiler_metrics(const string& annotation, size_t num_tiles) {
    stringstream ss;
    ss << "tile: " << num_tiles << endl;
    log_to_file(annotation, ss.str());
}

/**
 Helper function to log matrix load metrics
 */
inline void log_load_X_metrics(const string& annotation, int rows, int cols, size_t nnz, double load_time_ms) {
    stringstream ss;
    ss << "rows_X: " << rows << ", cols_X: " << cols << ", nnz_X: " << nnz << endl;
    ss << fixed << setprecision(3);
    ss << "disk to memory time: X load: " << load_time_ms << "ms" << endl;
    log_to_file(annotation, ss.str());
}

/**
 Helper function to log W matrix load metrics
 */
inline void log_load_W_metrics(const string& annotation, int rows, int cols, double load_time_ms) {
    stringstream ss;
    ss << "rows_W: " << rows << ", cols_W: " << cols << endl;
    ss << fixed << setprecision(3);
    ss << "disk to memory time: W load: " << load_time_ms << "ms" << endl;
    log_to_file(annotation, ss.str());
}

/**
 Helper function to log tile density classification metrics
 
 @param annotation Log file annotation
 @param num_dense Number of dense tiles
 @param num_sparse Number of sparse tiles
 */
inline void log_tile_density_metrics(const string& annotation, size_t num_dense, size_t num_sparse) {
    stringstream ss;
    ss << "dense_tiles: " << num_dense << ", sparse_tiles: " << num_sparse << endl;
    log_to_file(annotation, ss.str());
}

/**
 Helper function to log matrix density
 
 @param annotation Log file annotation
 @param matrix_density Overall density of the matrix (nnz / (rows * cols))
 */
inline void log_matrix_density(const string& annotation, double matrix_density) {
    stringstream ss;
    ss << fixed << setprecision(6);
    ss << "matrix_density: " << matrix_density << endl;
    log_to_file(annotation, ss.str());
}

/**
 Helper function to log OpenMP thread information
 
 @param annotation Log file annotation
 @param num_threads Number of OpenMP threads
 */
inline void log_openmp_threads(const string& annotation, int num_threads) {
    stringstream ss;
    ss << "OpenMP threads: " << num_threads << endl;
    log_to_file(annotation, ss.str());
}

/**
 Helper function to get log file path for tilepredpermspmm logs
 
 @param annotation Log file annotation (e.g., "2" for "2_tilepredpermspmm.txt")
 @param base_path Base directory for logs (default: "../logs/")
 @return Full path to log file
 */
inline string log_file_path_tilepredpermspmm(const string& annotation, const string& base_path = "../logs/") {
    if (annotation.empty()) {
        return base_path + "0_tilepredpermspmm.txt";
    }
    return base_path + annotation + "_tilepredpermspmm.txt";
}

/**
 Helper function to reset log file for tilepredpermspmm
 */
inline void reset_log_tilepredpermspmm(const string& annotation, const string& base_path = "../logs/") {
    string log_filename = log_file_path_tilepredpermspmm(annotation, base_path);
    fs::create_directories(fs::path(log_filename).parent_path());
    ofstream log(log_filename, ios::trunc);
}

/**
 Helper function to log to tilepredpermspmm log file
 */
inline void log_to_file_tilepredpermspmm(const string& annotation, const string& message, const string& base_path = "../logs/") {
    string log_filename = log_file_path_tilepredpermspmm(annotation, base_path);
    fs::create_directories(fs::path(log_filename).parent_path());
    ofstream log(log_filename, ios::app);
    if (log.is_open()) {
        log << message;
        log.close();
    }
}

/**
 Helper function to log OpenMP thread information to tilepredpermspmm log file
 
 @param annotation Log file annotation
 @param num_threads Number of OpenMP threads
 */
inline void log_openmp_threads_tilepredpermspmm(const string& annotation, int num_threads) {
    stringstream ss;
    ss << "OpenMP threads: " << num_threads << endl;
    log_to_file_tilepredpermspmm(annotation, ss.str());
}

#ifdef USE_CUDA
#include "dense_spmm_cuda.hpp"

/**
 Helper function to log CUDA device information to tilepredpermspmm log file
 
 @param annotation Log file annotation
 @param info CUDA device information
 */
inline void log_cuda_device_info_tilepredpermspmm(const string& annotation, const CudaDeviceInfo& info) {
    stringstream ss;
    if (info.available) {
        ss << "CUDA device: " << info.device_name 
           << " (compute capability " << info.compute_major << "." << info.compute_minor << ")" << endl;
        ss << "CUDA runtime: " << info.cuda_runtime_version << endl;
        ss << "cuBLAS version: " << info.cublas_version << endl;
        ss << "CUDA device ID: " << info.device_id << endl;
        ss << "GPU memory: " << info.total_memory_mb << " MB" << endl;
    } else {
        ss << "CUDA: not available" << endl;
    }
    log_to_file_tilepredpermspmm(annotation, ss.str());
}

/**
 Helper function to log CUDA usage statistics to tilepredpermspmm log file
 
 @param annotation Log file annotation
 @param cuda_tiles Number of dense tiles processed with CUDA
 @param cpu_tiles Number of dense tiles processed with CPU fallback
 */
inline void log_cuda_usage_stats_tilepredpermspmm(const string& annotation, 
                                                   size_t cuda_tiles, size_t cpu_tiles) {
    stringstream ss;
    ss << "CUDA dense tiles: " << cuda_tiles << endl;
    ss << "CPU dense tiles: " << cpu_tiles << endl;
    log_to_file_tilepredpermspmm(annotation, ss.str());
}
#endif // USE_CUDA

