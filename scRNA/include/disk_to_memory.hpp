#pragma once
#include "csr.hpp"
#include <string>
#include <vector>
using namespace std;

// Load X
// @param x_h5_path Path to HDF5 file containing X matrix
// @param log_annotation Optional log file annotation (e.g., "0" for log0.txt). If empty, no logging is performed.
CSR load_X_h5_as_csr(const string& x_h5_path, const string& log_annotation = "");

// Load W
// @param w_h5_path Path to HDF5 file containing W matrix
// @param nrows Output parameter for number of rows in W
// @param k Output parameter for number of columns in W
// @param log_annotation Optional log file annotation (e.g., "0" for log0.txt). If empty, no logging is performed.
vector<float> load_W_h5(const string& w_h5_path, int& nrows, int& k, const string& log_annotation = "");