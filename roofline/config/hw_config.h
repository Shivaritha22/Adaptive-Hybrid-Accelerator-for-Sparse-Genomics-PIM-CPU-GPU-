#pragma once

/*
 Hardware Configuration
  
  Defines fixed hardware parameters that are compile-time constants.
  These values represent the physical characteristics of the target hardware
  and should only be changed by editing this file and recompiling.
 */

namespace hw_config {
    // Tile dimensions for 2D tiling
    constexpr int TILE_ROWS = 64;   // Number of rows per tile
    constexpr int TILE_COLS = 64;   // Number of columns per tile
    
    // Thread configuration
    constexpr int NUM_THREADS = 8;  // Number of threads for parallel execution
    
    // Tile density threshold for dense/sparse classification
    // Tiles with density >= DENSE_TILE_THRESHOLD are classified as dense, else sparse
    constexpr double DENSE_TILE_THRESHOLD = 0.05;  
}

