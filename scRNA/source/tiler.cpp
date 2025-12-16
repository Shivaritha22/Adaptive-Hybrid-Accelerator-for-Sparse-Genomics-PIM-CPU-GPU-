#include "../include/tiler.hpp"
#include "../include/logger.hpp"
#include <algorithm>
#include <cmath>

using namespace std;

vector<Tile> make_2d_tiles(const CSR& X_prime, const TilingConfig& cfg, const string& log_annotation) {
    int n_rows = X_prime.nrows;
    int n_cols = X_prime.ncols;
    int T_R = cfg.tile_rows;
    int T_C = cfg.tile_cols;
    
    // Calculate number of tiles in each dimension
    int num_row_tiles = static_cast<int>(ceil(static_cast<double>(n_rows) / T_R));
    int num_col_tiles = static_cast<int>(ceil(static_cast<double>(n_cols) / T_C));
    
    // Initialize all tiles with zero nnz
    vector<Tile> tiles;
    tiles.reserve(num_row_tiles * num_col_tiles);
    
    for (int rb = 0; rb < num_row_tiles; rb++) {
        for (int cb = 0; cb < num_col_tiles; cb++) {
            int row_start = rb * T_R;
            int row_end = min(row_start + T_R, n_rows);
            int col_start = cb * T_C;
            int col_end = min(col_start + T_C, n_cols);
            
            Tile tile;
            tile.row_start = row_start;
            tile.row_end = row_end;
            tile.col_start = col_start;
            tile.col_end = col_end;
            tile.nnz = 0;
            tile.is_dense = false;  // Will be set by predictor
            
            tiles.push_back(tile);
        }
    }
    
    // Scan CSR and count nnz per tile
    for (int i = 0; i < n_rows; i++) {
        int row_start_idx = X_prime.indptr[i];
        int row_end_idx = X_prime.indptr[i + 1];
        
        for (int idx = row_start_idx; idx < row_end_idx; idx++) {
            int j = X_prime.indices[idx];
            
            // Determine which tile this nonzero belongs to
            int rb = i / T_R;
            int cb = j / T_C;
            
            // Bounds check (should always be valid, but check for safety)
            if (rb >= 0 && rb < num_row_tiles && cb >= 0 && cb < num_col_tiles) {
                int tile_idx = rb * num_col_tiles + cb;
                tiles[tile_idx].nnz++;
            }
        }
    }
    
    // Log number of tiles if annotation is provided
    if (!log_annotation.empty()) {
        log_tiler_metrics(log_annotation, tiles.size());
    }
    
    return tiles;
}

double tile_density(const Tile& t) {
    return t.density();
}

pair<size_t, size_t> predict_tile_density(vector<Tile>& tiles, double threshold) {
    size_t dense_count = 0;
    size_t sparse_count = 0;
    
    for (auto& tile : tiles) {
        double density = tile.density();
        tile.is_dense = (density >= threshold);
        
        if (tile.is_dense) {
            dense_count++;
        } else {
            sparse_count++;
        }
    }
    
    return make_pair(dense_count, sparse_count);
}

