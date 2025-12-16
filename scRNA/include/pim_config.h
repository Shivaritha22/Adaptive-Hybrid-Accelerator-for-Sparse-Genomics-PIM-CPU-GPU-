#pragma once

/*
  PIM Configuration 
 Defines modes and parameters for PIM processing stages.
 Shared by both baseline and PIM paths.
 */

enum class FilterMode {
    None,           // No filtering
    ValueThreshold  // Filter by absolute value threshold
    // Future: TopKPerRow, KeepFracPerRow
};

enum class QuantMode {
    None,           // No quantization
    Int8PerRow,     // Quantize to int8 per row
    Int8Global      // Quantize to int8 globally
    // Future: other quantization modes
};

struct PIMParams {
    FilterMode filter_mode = FilterMode::None;
    double value_threshold = 0.0;      // Manual threshold (if > 0, overrides auto-selection)
    double keep_frac_global = 0.5;     // Fraction of values to keep globally (default: top 50%)
    QuantMode quant_mode = QuantMode::None;
    // Future: other parameters for quantization, format changes, etc.
};

