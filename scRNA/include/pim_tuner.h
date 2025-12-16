#pragma once
#include "csr.hpp"
#include "pim_config.h"

/*
 * PIM Tuner Module
 * 
 * Automatic threshold selection and parameter tuning.
 * Analyzes dataset characteristics to choose optimal parameters.
 */

/**
 * Automatically select a value threshold based on dataset characteristics.
 * 
 * Algorithm: Global percentile-based thresholding
 * - Collects absolute values of all nonzeros in X
 * - Uses params.keep_frac_global (default 0.5) to determine threshold
 * - Computes k = floor((1 - keep_frac_global) * nnz)
 * - Finds k-th smallest absolute value using nth_element
 * - Returns that value as threshold
 * 
 * Intuition: keep_frac_global = 0.5 means keep top 50% largest values,
 *            dropping the smallest 50% by magnitude.
 * 
 * @param X Input CSR matrix
 * @param params PIM parameters (if value_threshold > 0, returns it directly;
 *               otherwise uses keep_frac_global for auto-selection)
 * @return Selected threshold value
 */
double auto_threshold_value(const CSR& X, const PIMParams& params);

