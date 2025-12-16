#pragma once
#include "csr.hpp"
#include "tiler.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>

using namespace std;

/**
 * Sparse-Dense Matrix Multiplication: Y = X * W
 * Pure computation function - no timing or logging.
 * @param log_annotation Optional log file annotation for OpenMP thread logging (e.g., "0" for log0.txt). If empty, no logging is performed.
 */
vector<float> spmm_baseline(const CSR& X, const vector<float>& W, int W_rows, int W_cols, const string& log_annotation = "");

/**
 * Tiled SpMM: Y = X * W using 2D tiling
 * Processes matrix tile by tile (PIM OFF)
 * 
 * Each tile processes only the nonzeros that fall within its row and column range.
 * This ensures each nonzero is processed exactly once across all tiles, matching
 * the baseline result.
 * 
 * @param X Sparse CSR matrix
 * @param W Dense weight matrix (row-major)
 * @param W_rows Number of rows in W
 * @param W_cols Number of columns in W
 * @param cfg Tiling configuration
 * @param log_annotation Optional log file annotation (e.g., "0" for log0.txt). If empty, no logging is performed.
 * @return Pair of (result vector Y, number of tiles used)
 */
pair<vector<float>, size_t> spmm_tiled(const CSR& X, const vector<float>& W, int W_rows, int W_cols, 
                                       const TilingConfig& cfg, const string& log_annotation = "");