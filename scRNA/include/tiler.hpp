#pragma once
#include "csr.hpp"
#include "../config/hw_config.h"
#include <vector>
#include <cstddef>
#include <string>
#include <utility>

using namespace std;

/*
 * 2D Tiling Module
 * 
 * Library for creating 2D tiles over a CSR matrix.
 * This module only handles tile creation and metadata - it does not
 * perform PIM filtering, quantization, or SpMM computation.
 */

/**
 * Tile struct: metadata describing one rectangular region of a matrix.
 * 
 * Stores the row/column bounds and nnz count for a tile.
 * Does not own any data - always interpreted in context of existing CSR storage.
 */
struct Tile {
    int row_start;      // Inclusive start row index
    int row_end;        // Exclusive end row index
    int col_start;      // Inclusive start column index
    int col_end;        // Exclusive end column index
    size_t nnz;         // Number of nonzeros in this tile
    bool is_dense;      // Classification: true if dense, false if sparse
    
    // Optional: compute density on demand
    double density() const {
        int rows = row_end - row_start;
        int cols = col_end - col_start;
        if (rows == 0 || cols == 0) return 0.0;
        return static_cast<double>(nnz) / (static_cast<double>(rows) * static_cast<double>(cols));
    }
    
    // Default constructor
    Tile() : row_start(0), row_end(0), col_start(0), col_end(0), nnz(0), is_dense(false) {}
};

/**
 * TilingConfig: configuration for 2D tiling.
 * 
 * Contains tile dimensions and optional references to permutation arrays.
 */
struct TilingConfig {
    int tile_rows;      // Number of rows per tile
    int tile_cols;      // Number of columns per tile
    
    // Optional: permutation arrays (nullptr if no permutation)
    const vector<int>* perm_r = nullptr;      // Row permutation: new_row = perm_r[old_row]
    const vector<int>* inv_perm_r = nullptr;  // Inverse row permutation: old_row = inv_perm_r[new_row]
    const vector<int>* perm_c = nullptr;      // Column permutation: new_col = perm_c[old_col]
    const vector<int>* inv_perm_c = nullptr;  // Inverse column permutation: old_col = inv_perm_c[new_col]
    
    // Constructor with default values from hw_config
    TilingConfig() 
        : tile_rows(hw_config::TILE_ROWS), 
          tile_cols(hw_config::TILE_COLS) {}
    
    // Constructor with custom tile sizes
    TilingConfig(int tr, int tc) 
        : tile_rows(tr), 
          tile_cols(tc) {}
};

/**
 * Create 2D tiles over a CSR matrix.
 * 
 * Divides the matrix into a grid of tiles with dimensions tile_rows x tile_cols.
 * Handles edge tiles correctly for non-square matrices.
 * 
 * @param X_prime The (possibly filtered and permuted) CSR matrix to tile
 * @param cfg Tiling configuration
 * @param log_annotation Optional log file annotation (e.g., "0" for log0.txt). If empty, no logging is performed.
 * @return Vector of Tile objects, one per tile in the grid
 */
vector<Tile> make_2d_tiles(const CSR& X_prime, const TilingConfig& cfg, const string& log_annotation = "");

/**
 * Helper function to compute tile density.
 * 
 * @param t Tile to compute density for
 * @return Density (nnz / (rows * cols))
 */
double tile_density(const Tile& t);

/**
 * Tile Density Predictor
 * 
 * Classifies tiles as dense or sparse based on their density.
 * For each tile: density = nnz / (tile_rows * tile_cols)
 * If density >= DENSE_TILE_THRESHOLD → mark as dense, else sparse.
 * 
 * @param tiles Vector of tiles to classify (modified in-place)
 * @param threshold Density threshold for classification (defaults to hw_config::DENSE_TILE_THRESHOLD)
 * @return Pair of (number of dense tiles, number of sparse tiles)
 */
pair<size_t, size_t> predict_tile_density(vector<Tile>& tiles, double threshold = hw_config::DENSE_TILE_THRESHOLD);

