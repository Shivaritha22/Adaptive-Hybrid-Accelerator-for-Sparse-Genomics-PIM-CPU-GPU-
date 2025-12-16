#include "../include/permutation.hpp"
#include "../include/disk_to_memory.hpp"
#include "../include/spmm.hpp"
#include "../include/tiler.hpp"
#include "../include/tile_spmm.hpp"
#include "../include/csr.hpp"
#include "../include/logger.hpp"
#include "../config/hw_config.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <H5Cpp.h>
#include <filesystem>
#include <chrono>

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

/**
 * Count mismatches between two matrices
 */
size_t count_mismatches(const vector<float>& Y1, const vector<float>& Y2, int rows, int cols) {
    if (Y1.size() != Y2.size() || Y1.size() != static_cast<size_t>(rows * cols)) {
        return Y1.size();  // Return max if dimensions don't match
    }
    
    size_t mismatches = 0;
    for (size_t i = 0; i < Y1.size(); i++) {
        if (!approx_equal(Y1[i], Y2[i])) {
            mismatches++;
        }
    }
    return mismatches;
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
    
    cout << "=== CUDA Tiled SpMM Test (run5) ===" << endl;
    #ifdef USE_CUDA
    cout << "CUDA: ENABLED" << endl;
    #else
    cout << "CUDA: DISABLED (using CPU fallback)" << endl;
    #endif
    cout << endl;
    
    try {
        string x_path = "../dataset/X/" + x_filename;
        string w_path = "../dataset/W/" + w_filename;
        string postfix = extract_postfix(x_filename);
        
        // Initialize log file
        reset_log_tilepredpermspmm(postfix);
        
        // Load X with timing and logging
        auto start_X = chrono::high_resolution_clock::now();
        CSR X_original = load_X_h5_as_csr(x_path, "");
        auto end_X = chrono::high_resolution_clock::now();
        double X_load_time_ms = chrono::duration_cast<chrono::microseconds>(end_X - start_X).count() / 1000.0;
        
        // Load W with timing and logging
        auto start_W = chrono::high_resolution_clock::now();
        int W_rows, W_cols;
        vector<float> W_original = load_W_h5(w_path, W_rows, W_cols, "");
        auto end_W = chrono::high_resolution_clock::now();
        double W_load_time_ms = chrono::duration_cast<chrono::microseconds>(end_W - start_W).count() / 1000.0;
        
        // Log load metrics
        stringstream ss;
        ss << "rows_X: " << X_original.nrows << ", cols_X: " << X_original.ncols 
           << ", nnz_X: " << X_original.nnz << endl;
        ss << fixed << setprecision(3);
        ss << "disk to memory time: X load: " << X_load_time_ms << "ms" << endl;
        ss << "rows_W: " << W_rows << ", cols_W: " << W_cols << endl;
        ss << "disk to memory time: W load: " << W_load_time_ms << "ms" << endl;
        log_to_file_tilepredpermspmm(postfix, ss.str());
        
        // Verify dimensions
        if (X_original.ncols != W_rows) {
            cerr << "Dimension mismatch: X.ncols (" << X_original.ncols 
                 << ") != W.rows (" << W_rows << ")" << endl;
            return 1;
        }
        
        // ============================================================
        // Step 1: Tile original X
        // ============================================================
        TilingConfig cfg;
        vector<Tile> tiles = make_2d_tiles(X_original, cfg, "");
        
        // ============================================================
        // Step 2: Compute tile densities and classify
        // ============================================================
        auto density_counts = predict_tile_density(tiles, hw_config::DENSE_TILE_THRESHOLD);
        size_t num_dense = density_counts.first;
        size_t num_sparse = density_counts.second;
        
        // Log tile metrics
        stringstream ss2;
        ss2 << "tile: " << tiles.size() << endl;
        ss2 << "dense_tiles: " << num_dense << ", sparse_tiles: " << num_sparse << endl;
        log_to_file_tilepredpermspmm(postfix, ss2.str());
        
        // Print output with labels
        cout << "tiles: " << tiles.size() << endl;
        cout << "dense: " << num_dense << endl;
        cout << "sparse: " << num_sparse << endl;
        
        // ============================================================
        // Step 3: Process all tiles (function handles accumulation and logging)
        // This will use CUDA for dense tiles if USE_CUDA is defined
        // ============================================================
        int Y_rows = X_original.nrows;
        int Y_cols = W_cols;
        vector<float> Y_final = process_tiles_with_predictor(X_original, W_original, W_rows, W_cols, 
                                                             tiles, postfix);
        
        // ============================================================
        // Step 4: Save result
        // ============================================================
        string y_output_path = "../dataset/Y/y" + postfix + "_cuda.h5";
        save_Y_h5(Y_final, Y_rows, Y_cols, y_output_path);
        cout << "Saved result to: " << y_output_path << endl;
        
        // ============================================================
// Step 5: Compare with Y_check (reference correctness run)
// ============================================================
string y_check_path = "../dataset/Y/y" + postfix + "_check.h5";
int y_check_rows, y_check_cols;
vector<float> Y_check = load_Y_h5(y_check_path, y_check_rows, y_check_cols);

if (Y_rows != y_check_rows || Y_cols != y_check_cols) {
    cerr << "Dimension mismatch with Y_check" << endl;
    return 1;
}

size_t mismatches = count_mismatches(Y_final, Y_check, Y_rows, Y_cols);
if (mismatches == 0) {
    cout << "✓ Y matches Y_check!" << endl;
} else {
    cout << "✗ Y mismatches vs Y_check: " << mismatches << " elements" << endl;
}

cout << "spmm done" << endl;

return (mismatches == 0) ? 0 : 1;

        
    } catch (const exception& e) {
        cerr << "  ✗ Error: " << e.what() << endl;
        return 1;
    }
}