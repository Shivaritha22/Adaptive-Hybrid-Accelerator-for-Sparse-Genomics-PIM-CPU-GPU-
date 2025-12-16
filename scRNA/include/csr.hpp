#pragma once
#include <vector>

using namespace std;

/*
 Standard Compressed Sparse Row (CSR) matrix.
 
   nrows  : number of rows
   ncols  : number of columns
   nnz    : number of stored nonzeros
 
   indptr : length = nrows + 1
   indptr[r]..indptr[r+1]-1 are the entries of row r

   indices: column indices of each stored nonzero
   data   : values of each stored nonzero
 */

struct CSR {
    int nrows = 0;
    int ncols = 0;
    size_t nnz = 0;

    vector<int>   indptr;    // size nrows+1
    vector<int>   indices;   // size nnz
    vector<float> data;      // size nnz
};
