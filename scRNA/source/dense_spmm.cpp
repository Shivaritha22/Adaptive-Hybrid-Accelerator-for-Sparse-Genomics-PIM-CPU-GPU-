#include "../include/dense_spmm.hpp"

using namespace std;

/*
  Dense Tile SpMM Processing: Processes all nonzeros within the tile's row and column range
 */
void dense_spmm_tile(const CSR& X, const vector<float>& W, int W_cols,
                     const Tile& tile, vector<float>& Y, int Y_cols) {
    // For each row in this dense tile
    for (int i = tile.row_start; i < tile.row_end; i++) {
        int row_start = X.indptr[i];
        int row_end = X.indptr[i + 1];
        
        // For each nonzero in this row, but only process those within tile's column range
        for (int idx = row_start; idx < row_end; idx++) {
            int k = X.indices[idx];
            
            // Only process nonzeros that fall within this tile's column range
            if (k >= tile.col_start && k < tile.col_end) {
                float x_val = X.data[idx];
                
                // Accumulate into Y (partial SpMM computation for this dense tile)
                for (int j = 0; j < W_cols; j++) {
                    Y[i * Y_cols + j] += x_val * W[k * W_cols + j];
                }
            }
        }
    }
}

