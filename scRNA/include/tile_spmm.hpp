#pragma once
#include "csr.hpp"
#include "tiler.hpp"
#include "permutation.hpp"
#include <vector>

using namespace std;

/**
 * Extract a tile from a CSR matrix as a standalone CSR matrix.
 * Column indices are remapped to be 0-based within the tile.
 * 
 * @param X Original CSR matrix
 * @param tile Tile metadata specifying the region to extract
 * @return Standalone CSR matrix representing the tile
 */
CSR extract_tile_csr(const CSR& X, const Tile& tile);

/**
 * Extract corresponding W rows for a tile.
 * 
 * @param W Original weight matrix (row-major)
 * @param W_rows Number of rows in W
 * @param W_cols Number of columns in W
 * @param tile Tile metadata (uses col_start and col_end to determine which W rows to extract)
 * @return Extracted W rows as a standalone matrix (row-major)
 */
vector<float> extract_tile_W(const vector<float>& W, int W_rows, int W_cols, const Tile& tile);

/**
 * Materialize a CSR tile to a dense matrix buffer.
 * 
 * @param X_tile CSR tile (0-based column indices)
 * @return Dense matrix (row-major, M_tile × K_tile)
 */
vector<float> materialize_csr_to_dense(const CSR& X_tile);

/**
 * Permute dense matrix rows.
 * 
 * @param X_dense Dense matrix (row-major, M × K)
 * @param M Number of rows
 * @param K Number of columns
 * @param row_new2old Mapping: row_new2old[new_row] = old_row
 * @return Permuted dense matrix (row-major, M × K)
 */
vector<float> permute_dense_rows(const vector<float>& X_dense, int M, int K,
                                const vector<int>& row_new2old);

/**
 * Permute dense matrix columns.
 * 
 * @param X_dense Dense matrix (row-major, M × K)
 * @param M Number of rows
 * @param K Number of columns
 * @param col_new2old Mapping: col_new2old[new_col] = old_col
 * @return Permuted dense matrix (row-major, M × K)
 */
vector<float> permute_dense_cols(const vector<float>& X_dense, int M, int K,
                                const vector<int>& col_new2old);

/**
 * CPU fallback for dense GEMM: Y = X * W
 * 
 * @param X_dense Dense matrix X (row-major, M × K)
 * @param W_dense Dense matrix W (row-major, K × N)
 * @param M Number of rows in X and Y
 * @param K Number of columns in X and rows in W
 * @param N Number of columns in W and Y
 * @return Dense matrix Y (row-major, M × N)
 */
vector<float> dense_spmm_cpu_tile(const float* X_dense, const float* W_dense,
                                  int M, int K, int N);

/**
 * Dense tile SpMM with permutation workflow.
 * This function:
 * (a) Converts CSR tile to dense M×K buffer
 * (b) Applies row/column permutation to dense tile and/or W slice
 * (c) Calls either CUDA or CPU GEMM
 * (d) Unpermutes the Y rows
 * 
 * @param X_tile Extracted tile as standalone CSR (0-based column indices)
 * @param W_tile Extracted W rows as standalone matrix (row-major)
 * @param W_tile_rows Number of rows in W_tile (should equal tile.ncols)
 * @param W_cols Number of columns in W
 * @return Result matrix Y_tile (row-major) with permuted rows unpermuted
 */
vector<float> dense_perm_spmm_tile(const CSR& X_tile, const vector<float>& W_tile, 
                                    int W_tile_rows, int W_cols);

/**
 * Sparse tile SpMM without permutation.
 * Direct SpMM computation on a tile.
 * 
 * @param X_tile Extracted tile as standalone CSR (0-based column indices)
 * @param W_tile Extracted W rows as standalone matrix (row-major)
 * @param W_tile_rows Number of rows in W_tile (should equal tile.ncols)
 * @param W_cols Number of columns in W
 * @return Result matrix Y_tile (row-major)
 */
vector<float> sparse_spmm_tile(const CSR& X_tile, const vector<float>& W_tile, 
                               int W_tile_rows, int W_cols);

/**
 * Process all tiles with predictor-based routing and accumulate metrics.
 * This function handles the entire tiled SpMM workflow with logging.
 * 
 * @param X_original Original CSR matrix
 * @param W_original Original weight matrix (row-major)
 * @param W_rows Number of rows in W
 * @param W_cols Number of columns in W
 * @param tiles Vector of tiles to process
 * @param log_annotation Log file annotation (e.g., "2" for "2_tilepredpermspmm.txt")
 * @return Result matrix Y (row-major)
 */
vector<float> process_tiles_with_predictor(const CSR& X_original, 
                                          const vector<float>& W_original,
                                          int W_rows, int W_cols,
                                          const vector<Tile>& tiles,
                                          const string& log_annotation);

