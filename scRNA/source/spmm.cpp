#include "../include/spmm.hpp"
#include "../include/tiler.hpp"
#include "../include/dense_spmm.hpp"
#include "../include/logger.hpp"
#include <stdexcept>

using namespace std;

/*
  Tiled SpMM: Y = X * W using 2D tiling
  Processes matrix tile by tile (PIM OFF)
  Each tile processes only the nonzeros that fall within its row and column range.
  This ensures each nonzero is processed exactly once across all tiles, matching the baseline result.
 
  @param X Sparse CSR matrix
  @param W Dense weight matrix (row-major)
  @param W_rows Number of rows in W
  @param W_cols Number of columns in W
  @param cfg Tiling configuration
  @param log_annotation Optional log file annotation (e.g., "0" for log0.txt). If empty, no logging is performed.
  @return Pair of (result vector Y, number of tiles used)
 */
pair<vector<float>, size_t> spmm_tiled(const CSR& X, const vector<float>& W, int W_rows, int W_cols, 
                                       const TilingConfig& cfg, const string& log_annotation) {
    if (X.ncols != W_rows) {
        throw runtime_error("Matrix dimension mismatch: X.ncols=" + to_string(X.ncols) 
                           + " != W.nrows=" + to_string(W_rows));
    }
    
    int Y_rows = X.nrows;
    int Y_cols = W_cols;
    vector<float> Y(Y_rows * Y_cols, 0.0f);
    
    // Create tiles 
    vector<Tile> tiles = make_2d_tiles(X, cfg, log_annotation);
    size_t num_tiles = tiles.size();
    
    // Predict tile density and classify as dense/sparse
    auto density_counts = predict_tile_density(tiles);
    size_t num_dense = density_counts.first;
    size_t num_sparse = density_counts.second;
    
    // Log tile density metrics and matrix density
    if (!log_annotation.empty()) {
        log_tile_density_metrics(log_annotation, num_dense, num_sparse);
        
        // Calculate and log overall matrix density
        double matrix_density = 0.0;
        if (X.nrows > 0 && X.ncols > 0) {
            matrix_density = static_cast<double>(X.nnz) / (static_cast<double>(X.nrows) * static_cast<double>(X.ncols));
        }
        log_matrix_density(log_annotation, matrix_density);
    }
    
    // Process each tile: route based on density classification
    for (const auto& tile : tiles) {
        if (tile.is_dense) {
            // Dense tile: route to dense_spmm_tile function
            dense_spmm_tile(X, W, W_cols, tile, Y, Y_cols);
        } else {
            // Sparse tile: 
            for (int i = tile.row_start; i < tile.row_end; i++) {
                int row_start = X.indptr[i];
                int row_end = X.indptr[i + 1];
                
                // For each nonzero in this row, but only process those within tile's column range
                for (int idx = row_start; idx < row_end; idx++) {
                    int k = X.indices[idx];
                    
                    // Only process nonzeros that fall within this tile's column range
                    if (k >= tile.col_start && k < tile.col_end) {
                        float x_val = X.data[idx];
                        
                        // Accumulate into Y (partial SpMM computation for this sparse tile)
                        for (int j = 0; j < W_cols; j++) {
                            Y[i * Y_cols + j] += x_val * W[k * W_cols + j];
                        }
                    }
                }
            }
        }
    }
    
    return make_pair(Y, num_tiles);
}

