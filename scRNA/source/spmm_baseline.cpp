#include "../include/spmm.hpp"
#include "../include/logger.hpp"
#include <stdexcept>
#include <omp.h>

using namespace std;

/**
 Sparse-Dense Matrix Multiplication (Baseline): Y = X * W
 Pure computation function - no timing or logging.
 */
vector<float> spmm_baseline(const CSR& X, const vector<float>& W, int W_rows, int W_cols, const string& log_annotation) {
    if (X.ncols != W_rows) {
        throw runtime_error("Matrix dimension mismatch: X.ncols=" + to_string(X.ncols) 
                           + " != W.nrows=" + to_string(W_rows));
    }
    
    // Log OpenMP thread info
    if (!log_annotation.empty()) {
        int max_threads = omp_get_max_threads();
        log_openmp_threads(log_annotation, max_threads);
    }
    
    int Y_rows = X.nrows;
    int Y_cols = W_cols;
    vector<float> Y(Y_rows * Y_cols, 0.0f);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < X.nrows; i++) {
        int row_start = X.indptr[i];
        int row_end = X.indptr[i + 1];
        
        for (int idx = row_start; idx < row_end; idx++) {
            int k = X.indices[idx];
            float x_val = X.data[idx];
            
            for (int j = 0; j < W_cols; j++) {
                Y[i * Y_cols + j] += x_val * W[k * W_cols + j];
            }
        }
    }
    
    return Y;
}

