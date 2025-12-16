#pragma once
#include "csr.hpp"

/*
 * PIM Filter Module
 * 
 * Low-level PIM filtering kernels.
 * Pure filtering operations - no dataset stats, no auto-selection.
 */

/**
 * Filter CSR matrix by value threshold.
 * Drops all entries where |value| < threshold.
 * 
 * @param X Input CSR matrix
 * @param threshold Minimum absolute value to keep
 * @return New CSR matrix with filtered values
 */
CSR pim_filter_value_threshold(const CSR& X, double threshold);

// Future filtering functions:
// CSR pim_filter_topk_per_row(const CSR& X, int k);
// CSR pim_filter_keep_frac_per_row(const CSR& X, double frac);

