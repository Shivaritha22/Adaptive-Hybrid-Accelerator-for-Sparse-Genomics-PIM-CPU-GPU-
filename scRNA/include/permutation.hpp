#pragma once
#include "csr.hpp"
#include <vector>

/**
 Permutation Module - Row Permutation Only (Simplified)
 
 Uses "new2old" semantics: row_new2old[new_row] = old_row
 This means: new row i comes from original row row_new2old[i]
 */

/**
 Compute nnz per row for a CSR matrix.
 */
std::vector<size_t> compute_nnz_per_row(const CSR& X);

/**
 Create row permutation mapping (new → old).
 Rows are sorted by nnz (descending or ascending).
 
 @param nnz_per_row Vector of nnz counts per row
 @param descending If true, sort by descending nnz (high nnz first)
 @return Mapping: row_new2old[new_row] = old_row
 */
std::vector<int> create_row_new2old(const std::vector<size_t>& nnz_per_row, bool descending = true);

/**
 Permute CSR matrix rows only.
 
 @param X Original CSR matrix
 @param row_new2old Mapping: row_new2old[new_row] = old_row
 @return Permuted CSR matrix X' where X'[new_row, :] = X[old_row, :]
 */
CSR permute_csr_rows(const CSR& X, const std::vector<int>& row_new2old);

/**
 Permute weight matrix rows only.
 
 @param W Original weight matrix (row-major)
 @param W_rows Number of rows in W
 @param W_cols Number of columns in W
 @param row_new2old Mapping: row_new2old[new_row] = old_row
 @return Permuted weight matrix W' where W'[new_row, :] = W[old_row, :]
 */
std::vector<float> permute_weight_rows(const std::vector<float>& W,
                                       int W_rows,
                                       int W_cols,
                                       const std::vector<int>& row_new2old);

/**
 Unpermute CSR matrix rows (recover original row order).
 
 @param X_permuted Permuted CSR matrix
 @param row_new2old Mapping: row_new2old[new_row] = old_row
 @return Unpermuted CSR matrix X where X[old_row, :] = X_permuted[new_row, :]
 */
CSR unpermute_csr_rows(const CSR& X_permuted, const std::vector<int>& row_new2old);

/**
 Unpermute result matrix rows.
 
 @param Y_prime Permuted result matrix
 @param Y_rows Number of rows in Y
 @param Y_cols Number of columns in Y
 @param row_new2old Mapping: row_new2old[new_row] = old_row
 @return Unpermuted result matrix Y where Y[old_row, :] = Y'[new_row, :]
 */
std::vector<float> unpermute_rows(const std::vector<float>& Y_prime,
                                  int Y_rows,
                                  int Y_cols,
                                  const std::vector<int>& row_new2old);

/**
 Column Permutation Functions
 */

/**
 Compute nnz per column for a CSR matrix.
 */
std::vector<size_t> compute_nnz_per_col(const CSR& X);

/**
 Create column permutation mapping (new → old).
 Columns are sorted by nnz (descending or ascending).
 
 @param nnz_per_col Vector of nnz counts per column
 @param descending If true, sort by descending nnz (high nnz first)
 @return Mapping: col_new2old[new_col] = old_col
 */
std::vector<int> create_col_new2old(const std::vector<size_t>& nnz_per_col, bool descending = true);

/**
 Permute CSR matrix columns only.
 
 @param X Original CSR matrix
 @param col_new2old Mapping: col_new2old[new_col] = old_col
 @return Permuted CSR matrix X' where X'[:, new_col] = X[:, old_col]
 */
CSR permute_csr_cols(const CSR& X, const std::vector<int>& col_new2old);

/**
 Unpermute CSR matrix columns (recover original column order).
 
 @param X_permuted Permuted CSR matrix
 @param col_new2old Mapping: col_new2old[new_col] = old_col
 @return Unpermuted CSR matrix X where X[:, old_col] = X_permuted[:, new_col]
 */
CSR unpermute_csr_cols(const CSR& X_permuted, const std::vector<int>& col_new2old);
