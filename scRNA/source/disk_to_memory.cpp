#include "../include/disk_to_memory.hpp"
#include "../include/logger.hpp"
#include <H5Cpp.h>
#include <iostream>
#include <fstream> 
#include <chrono>  
#include <iomanip> 

using namespace std;
using namespace H5;

CSR load_X_h5_as_csr(const string& x_h5_path, const string& log_annotation) {
    auto start = chrono::high_resolution_clock::now(); // <--- Start Timer

    cout << "[disk_to_memory] Loading X from: " << x_h5_path << endl;

    CSR csr;
    
    try {
        H5File file(x_h5_path, H5F_ACC_RDONLY);
        Group matrix_group = file.openGroup("matrix");
        
        DataSet shape_dataset = matrix_group.openDataSet("shape");
        DataSpace shape_dataspace = shape_dataset.getSpace();
        hsize_t shape_dims[1];
        shape_dataspace.getSimpleExtentDims(shape_dims);
        vector<long long> shape(shape_dims[0]);
        shape_dataset.read(shape.data(), PredType::NATIVE_INT64);
        
        int n_genes = shape[0];
        int n_cells = shape[1];
        
        DataSet data_dataset = matrix_group.openDataSet("data");
        DataSpace data_dataspace = data_dataset.getSpace();
        hsize_t data_dims[1];
        data_dataspace.getSimpleExtentDims(data_dims);
        size_t nnz = data_dims[0];
        
        vector<float> data(nnz);
        data_dataset.read(data.data(), PredType::NATIVE_FLOAT);
        
        DataSet indices_dataset = matrix_group.openDataSet("indices");
        vector<int> indices(nnz);
        indices_dataset.read(indices.data(), PredType::NATIVE_INT32);
        
        DataSet indptr_dataset = matrix_group.openDataSet("indptr");
        DataSpace indptr_dataspace = indptr_dataset.getSpace();
        hsize_t indptr_dims[1];
        indptr_dataspace.getSimpleExtentDims(indptr_dims);
        vector<int> indptr(indptr_dims[0]);
        indptr_dataset.read(indptr.data(), PredType::NATIVE_INT32);
        
        cout << "[disk_to_memory] X shape: " << n_genes << " x " << n_cells 
             << ", nnz: " << nnz << endl;
        
        csr.nrows = n_genes;
        csr.ncols = n_cells;
        csr.nnz = nnz;
        csr.indptr.resize(n_genes + 1, 0);
        csr.indices.resize(nnz);
        csr.data.resize(nnz);
        
        for (size_t i = 0; i < nnz; i++) {
            csr.indptr[indices[i] + 1]++;
        }
        
        for (int i = 0; i < n_genes; i++) {
            csr.indptr[i + 1] += csr.indptr[i];
        }
        
        vector<int> row_counters = csr.indptr;
        for (int col = 0; col < n_cells; col++) {
            for (int idx = indptr[col]; idx < indptr[col + 1]; idx++) {
                int row = indices[idx];
                int dest = row_counters[row]++;
                csr.indices[dest] = col;
                csr.data[dest] = data[idx];
            }
        }
        
        cout << "[disk_to_memory] Successfully loaded and transposed X to CSR" << endl;
        
    } catch (Exception& e) {
        cerr << "[disk_to_memory] HDF5 error: " << e.getDetailMsg() << endl;
        throw;
    }

    
    auto end = chrono::high_resolution_clock::now();
    auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double duration_ms = duration_us / 1000.0;
    
    
    if (!log_annotation.empty()) {
        log_load_X_metrics(log_annotation, csr.nrows, csr.ncols, csr.nnz, duration_ms);
    }
    
    
    return csr;
}

vector<float> load_W_h5(const string& w_h5_path, int& nrows, int& k, const string& log_annotation) {
    auto start = chrono::high_resolution_clock::now(); // <--- Start Timer

    cout << "[disk_to_memory] Loading W from: " << w_h5_path << endl;
    
    vector<float> W_data;
    
    try {
        H5File file(w_h5_path, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet("W");
        DataSpace dataspace = dataset.getSpace();
        
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        nrows = dims[0];
        k = dims[1];
        
        W_data.resize(nrows * k);
        dataset.read(W_data.data(), PredType::NATIVE_FLOAT);
        
        cout << "[disk_to_memory] W shape: " << nrows << " x " << k << endl;
        
    } catch (Exception& e) {
        cerr << "[disk_to_memory] HDF5 error: " << e.getDetailMsg() << endl;
        throw;
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration_us = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double duration_ms = duration_us / 1000.0;
    
    
    if (!log_annotation.empty()) {
        log_load_W_metrics(log_annotation, nrows, k, duration_ms);
    }

    return W_data;
}