#pragma once
#include "csr.hpp"
#include "tiler.hpp"
#include <vector>

using namespace std;

/**
 Dense Tile SpMM Processing
 
  Performs normal SpMM computation for a single dense tile.
  This function processes the nonzeros within the tile's row and column range
  and accumulates the results into the output matrix Y.
  @param X Sparse CSR matrix
  @param W Dense weight matrix (row-major)
  @param W_cols Number of columns in W
  @param tile The dense tile to process
  @param Y Output matrix (row-major, modified in-place)
  @param Y_cols Number of columns in Y
 */
void dense_spmm_tile(const CSR& X, const vector<float>& W, int W_cols,
                     const Tile& tile, vector<float>& Y, int Y_cols);

