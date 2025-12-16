#pragma once
#include "../include/pim_config.h"

/*
  PIM Default Parameters
  
  Defines default parameters for PIM processing that can be overridden
  at runtime but have sensible defaults matching typical use cases.
  These are compile-time constants that serve as defaults.
 */

namespace pim_defaults {
    // Global filtering defaults
    constexpr double KEEP_FRAC_GLOBAL = 0.5;  // Default: keep top 50% of values
    
    // Tile density threshold for hybrid CPU/GPU scheduling
    // Tiles with density >= DENSE_TILE_THRESHOLD are considered dense
    constexpr double DENSE_TILE_THRESHOLD = 0.5;  // 50% density threshold
    
    // Default filter mode
    constexpr FilterMode DEFAULT_FILTER_MODE = FilterMode::ValueThreshold;
    
    // Default quantization mode
    constexpr QuantMode DEFAULT_QUANT_MODE = QuantMode::None;
}

