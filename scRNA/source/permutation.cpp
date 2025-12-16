#include "../include/permutation.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

using std::vector;
using std::size_t;

vector<size_t> compute_nnz_per_row(const CSR& X) {
    vector<size_t> nnz_per_row(X.nrows, 0);
    for (int i = 0; i < X.nrows; ++i) {
        nnz_per_row[i] = static_cast<size_t>(X.indptr[i + 1] - X.indptr[i]);
    }
    return nnz_per_row;
}

vector<int> create_row_new2old(const vector<size_t>& nnz_per_row, bool descending) {
    int n = static_cast<int>(nnz_per_row.size());
    vector<int> row_new2old(n);
    
    
    std::iota(row_new2old.begin(), row_new2old.end(), 0);
    
    
    if (descending) {
        std::sort(row_new2old.begin(), row_new2old.end(),
                  [&nnz_per_row](int a, int b) {
                      return nnz_per_row[a] > nnz_per_row[b];
                  });
    } else {
        std::sort(row_new2old.begin(), row_new2old.end(),
                  [&nnz_per_row](int a, int b) {
                      return nnz_per_row[a] < nnz_per_row[b];
                  });
    }
    
    
    return row_new2old;
}

CSR permute_csr_rows(const CSR& X, const vector<int>& row_new2old) {
    if (static_cast<int>(row_new2old.size()) != X.nrows) {
        throw std::runtime_error("permute_csr_rows: row_new2old size mismatch");
    }
    
    CSR Xp;
    Xp.nrows = X.nrows;
    Xp.ncols = X.ncols;
    Xp.nnz   = X.nnz;
    Xp.indptr.assign(Xp.nrows + 1, 0);
    Xp.indices.resize(X.nnz);
    Xp.data.resize(X.nnz);
    
    // First pass: count nnz per new row
    for (int i_new = 0; i_new < Xp.nrows; ++i_new) {
        int i_old = row_new2old[i_new];
        int row_start = X.indptr[i_old];
        int row_end   = X.indptr[i_old + 1];
        int row_nnz   = row_end - row_start;
        Xp.indptr[i_new + 1] = Xp.indptr[i_new] + row_nnz;
    }
    
    // Second pass: copy rows
    vector<int> write_ptr = Xp.indptr; 
    for (int i_new = 0; i_new < Xp.nrows; ++i_new) {
        int i_old = row_new2old[i_new];
        int row_start = X.indptr[i_old];
        int row_end   = X.indptr[i_old + 1];
        
        for (int idx = row_start; idx < row_end; ++idx) {
            int dest = write_ptr[i_new]++;
            Xp.indices[dest] = X.indices[idx];
            Xp.data[dest]    = X.data[idx];
        }
    }
    
    for (int i_new = 0; i_new < Xp.nrows; ++i_new) {
        int start = Xp.indptr[i_new];
        int end   = Xp.indptr[i_new + 1];
        vector<std::pair<int,float>> row_data;
        row_data.reserve(end - start);
        for (int idx = start; idx < end; ++idx) {
            row_data.emplace_back(Xp.indices[idx], Xp.data[idx]);
        }
        std::sort(row_data.begin(), row_data.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
        for (int k = 0; k < static_cast<int>(row_data.size()); ++k) {
            Xp.indices[start + k] = row_data[k].first;
            Xp.data[start + k]    = row_data[k].second;
        }
    }
    
    return Xp;
}

vector<float> permute_weight_rows(const vector<float>& W,
                                  int W_rows,
                                  int W_cols,
                                  const vector<int>& row_new2old) {
    if (static_cast<int>(row_new2old.size()) != W_rows) {
        throw std::runtime_error("permute_weight_rows: row_new2old size mismatch");
    }
    
    if (static_cast<int>(W.size()) != W_rows * W_cols) {
        throw std::runtime_error("permute_weight_rows: W size mismatch");
    }
    
    vector<float> Wp(W_rows * W_cols);
    
    for (int i_new = 0; i_new < W_rows; ++i_new) {
        int i_old = row_new2old[i_new];
        if (i_old < 0 || i_old >= W_rows) {
            throw std::runtime_error("permute_weight_rows: invalid row_new2old entry");
        }
        for (int j = 0; j < W_cols; ++j) {
            Wp[i_new * W_cols + j] = W[i_old * W_cols + j];
        }
    }
    
    return Wp;
}

/**
  Unpermute CSR matrix rows 
 */
CSR unpermute_csr_rows(const CSR& X_permuted, const vector<int>& row_new2old) {
    if (static_cast<int>(row_new2old.size()) != X_permuted.nrows) {
        throw std::runtime_error("unpermute_csr_rows: row_new2old size mismatch");
    }
    
    CSR X;
    X.nrows = X_permuted.nrows;
    X.ncols = X_permuted.ncols;
    X.nnz = X_permuted.nnz;
    X.indptr.assign(X.nrows + 1, 0);
    X.indices.resize(X.nnz);
    X.data.resize(X.nnz);
    

    // First pass: count nnz per old row
    for (int i_new = 0; i_new < X_permuted.nrows; ++i_new) {
        int i_old = row_new2old[i_new];
        if (i_old < 0 || i_old >= X.nrows) {
            throw std::runtime_error("unpermute_csr_rows: invalid row_new2old entry");
        }
        int row_start = X_permuted.indptr[i_new];
        int row_end = X_permuted.indptr[i_new + 1];
        int row_nnz = row_end - row_start;
        X.indptr[i_old + 1] += row_nnz;
    }
    
    // Build cumulative indptr
    for (int i = 0; i < X.nrows; ++i) {
        X.indptr[i + 1] += X.indptr[i];
    }
    
    // Second pass: copy rows to original positions
    vector<int> write_ptr = X.indptr;
    for (int i_new = 0; i_new < X_permuted.nrows; ++i_new) {
        int i_old = row_new2old[i_new];
        int row_start = X_permuted.indptr[i_new];
        int row_end = X_permuted.indptr[i_new + 1];
        
        for (int idx = row_start; idx < row_end; ++idx) {
            int dest = write_ptr[i_old]++;
            X.indices[dest] = X_permuted.indices[idx];
            X.data[dest] = X_permuted.data[idx];
        }
    }
    
    // Sort each row by column index
    for (int i = 0; i < X.nrows; ++i) {
        int start = X.indptr[i];
        int end = X.indptr[i + 1];
        vector<std::pair<int, float>> row_data;
        row_data.reserve(end - start);
        for (int idx = start; idx < end; ++idx) {
            row_data.emplace_back(X.indices[idx], X.data[idx]);
        }
        std::sort(row_data.begin(), row_data.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
        for (int k = 0; k < static_cast<int>(row_data.size()); ++k) {
            X.indices[start + k] = row_data[k].first;
            X.data[start + k] = row_data[k].second;
        }
    }
    
    return X;
}

vector<float> unpermute_rows(const vector<float>& Y_prime,
                             int Y_rows,
                             int Y_cols,
                             const vector<int>& row_new2old) {
    if (static_cast<int>(row_new2old.size()) != Y_rows) {
        throw std::runtime_error("unpermute_rows: row_new2old size mismatch");
    }
    
    if (static_cast<int>(Y_prime.size()) != Y_rows * Y_cols) {
        throw std::runtime_error("unpermute_rows: Y_prime size mismatch");
    }
    
    vector<float> Y(Y_rows * Y_cols);
    
    // row_new2old[new_row] = old_row
    // Y_prime[new_row, :] = Y_base[old_row, :]
    // So: Y[old_row, :] = Y_prime[new_row, :]
    for (int i_new = 0; i_new < Y_rows; ++i_new) {
        int i_old = row_new2old[i_new];
        if (i_old < 0 || i_old >= Y_rows) {
            throw std::runtime_error("unpermute_rows: invalid row_new2old entry");
        }
        for (int j = 0; j < Y_cols; ++j) {
            Y[i_old * Y_cols + j] = Y_prime[i_new * Y_cols + j];
        }
    }
    
    return Y;
}

/*
 Column Permutation Functions
 */

vector<size_t> compute_nnz_per_col(const CSR& X) {
    vector<size_t> nnz_per_col(X.ncols, 0);
    for (size_t i = 0; i < X.nnz; ++i) {
        int col = X.indices[i];
        if (col >= 0 && col < X.ncols) {
            nnz_per_col[col]++;
        }
    }
    return nnz_per_col;
}

vector<int> create_col_new2old(const vector<size_t>& nnz_per_col, bool descending) {
    int n = static_cast<int>(nnz_per_col.size());
    vector<int> col_new2old(n);
    
    std::iota(col_new2old.begin(), col_new2old.end(), 0);
    
    if (descending) {
        std::sort(col_new2old.begin(), col_new2old.end(),
                  [&nnz_per_col](int a, int b) {
                      return nnz_per_col[a] > nnz_per_col[b];
                  });
    } else {
        std::sort(col_new2old.begin(), col_new2old.end(),
                  [&nnz_per_col](int a, int b) {
                      return nnz_per_col[a] < nnz_per_col[b];
                  });
    }
    
    return col_new2old;
}

CSR permute_csr_cols(const CSR& X, const vector<int>& col_new2old) {
    if (static_cast<int>(col_new2old.size()) != X.ncols) {
        throw std::runtime_error("permute_csr_cols: col_new2old size mismatch");
    }
    

    vector<int> col_old2new(X.ncols);
    for (int new_col = 0; new_col < X.ncols; ++new_col) {
        int old_col = col_new2old[new_col];
        if (old_col < 0 || old_col >= X.ncols) {
            throw std::runtime_error("permute_csr_cols: invalid col_new2old entry");
        }
        col_old2new[old_col] = new_col;
    }
    
    CSR Xp;
    Xp.nrows = X.nrows;
    Xp.ncols = X.ncols;
    Xp.nnz = X.nnz;
    Xp.indptr = X.indptr;  
    Xp.indices.resize(X.nnz);
    Xp.data = X.data; 
    
    
    for (size_t i = 0; i < X.nnz; ++i) {
        int old_col = X.indices[i];
        if (old_col < 0 || old_col >= X.ncols) {
            throw std::runtime_error("permute_csr_cols: invalid column index in X");
        }
        Xp.indices[i] = col_old2new[old_col];
    }
    
    
    for (int i = 0; i < Xp.nrows; ++i) {
        int start = Xp.indptr[i];
        int end = Xp.indptr[i + 1];
        vector<std::pair<int, float>> row_data;
        row_data.reserve(end - start);
        for (int idx = start; idx < end; ++idx) {
            row_data.emplace_back(Xp.indices[idx], Xp.data[idx]);
        }
        std::sort(row_data.begin(), row_data.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
        for (int k = 0; k < static_cast<int>(row_data.size()); ++k) {
            Xp.indices[start + k] = row_data[k].first;
            Xp.data[start + k] = row_data[k].second;
        }
    }
    
    return Xp;
}

CSR unpermute_csr_cols(const CSR& X_permuted, const vector<int>& col_new2old) {
    if (static_cast<int>(col_new2old.size()) != X_permuted.ncols) {
        throw std::runtime_error("unpermute_csr_cols: col_new2old size mismatch");
    }
    
    vector<int> col_old2new(X_permuted.ncols);
    for (int new_col = 0; new_col < X_permuted.ncols; ++new_col) {
        int old_col = col_new2old[new_col];
        if (old_col < 0 || old_col >= X_permuted.ncols) {
            throw std::runtime_error("unpermute_csr_cols: invalid col_new2old entry");
        }
        col_old2new[old_col] = new_col;
    }
    
    CSR X;
    X.nrows = X_permuted.nrows;
    X.ncols = X_permuted.ncols;
    X.nnz = X_permuted.nnz;
    X.indptr = X_permuted.indptr;  // Row structure doesn't change
    X.indices.resize(X.nnz);
    X.data = X_permuted.data;  // Values don't change, only column indices
    
    // Remap column indices: new_col -> old_col
    // For each nonzero at (i, new_col) in X_permuted, set X[i, old_col] = X_permuted[i, new_col]
    // where old_col = col_new2old[new_col]
    for (size_t i = 0; i < X_permuted.nnz; ++i) {
        int new_col = X_permuted.indices[i];
        if (new_col < 0 || new_col >= X_permuted.ncols) {
            throw std::runtime_error("unpermute_csr_cols: invalid column index in X_permuted");
        }
        int old_col = col_new2old[new_col];
        X.indices[i] = old_col;
    }
    
    // Sort each row by column index to maintain CSR format
    for (int i = 0; i < X.nrows; ++i) {
        int start = X.indptr[i];
        int end = X.indptr[i + 1];
        vector<std::pair<int, float>> row_data;
        row_data.reserve(end - start);
        for (int idx = start; idx < end; ++idx) {
            row_data.emplace_back(X.indices[idx], X.data[idx]);
        }
        std::sort(row_data.begin(), row_data.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
        for (int k = 0; k < static_cast<int>(row_data.size()); ++k) {
            X.indices[start + k] = row_data[k].first;
            X.data[start + k] = row_data[k].second;
        }
    }
    
    return X;
}
