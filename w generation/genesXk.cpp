#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <stdexcept>

#include <hdf5.h>

using namespace std;

// Read [n_cells, n_features]
pair<hsize_t, hsize_t> read_matrix_shape(const string& x_h5_path) {
    hid_t file = H5Fopen(x_h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw runtime_error("Cannot open X file: " + x_h5_path);

    hid_t dset = H5Dopen2(file, "/matrix/shape", H5P_DEFAULT);
    if (dset < 0) {
        H5Fclose(file);
        throw runtime_error("Cannot open /matrix/shape in " + x_h5_path);
    }

    hid_t space = H5Dget_space(dset);
    hssize_t ndims = H5Sget_simple_extent_ndims(space);
    if (ndims != 1) {
        H5Sclose(space);
        H5Dclose(dset);
        H5Fclose(file);
        throw runtime_error("Unexpected rank for /matrix/shape");
    }

    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    if (dims[0] != 2) {
        H5Sclose(space);
        H5Dclose(dset);
        H5Fclose(file);
        throw runtime_error("Expected /matrix/shape to have length 2");
    }

    long long shape[2];
    herr_t status = H5Dread(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, shape);
    if (status < 0) {
        H5Sclose(space);
        H5Dclose(dset);
        H5Fclose(file);
        throw runtime_error("Failed to read /matrix/shape");
    }

    H5Sclose(space);
    H5Dclose(dset);
    H5Fclose(file);

    hsize_t n_cells  = static_cast<hsize_t>(shape[0]);
    hsize_t n_genes  = static_cast<hsize_t>(shape[1]);  // columns
    return {n_cells, n_genes};
}

// Write W (n_genes x k) as float32 dataset
void write_W_h5(const string& w_h5_path,
                const vector<float>& W,
                hsize_t n_genes,
                hsize_t k) {
    hid_t file = H5Fcreate(w_h5_path.c_str(), H5F_ACC_TRUNC,
                           H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) throw runtime_error("Cannot create W file: " + w_h5_path);

    hsize_t dims[2] = { n_genes, k };
    hid_t space = H5Screate_simple(2, dims, nullptr);
    if (space < 0) {
        H5Fclose(file);
        throw runtime_error("Failed to create dataspace for W");
    }

    hid_t dset = H5Dcreate2(file, "/W",
                            H5T_IEEE_F32LE,
                            space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        H5Sclose(space);
        H5Fclose(file);
        throw runtime_error("Failed to create dataset /W");
    }

    herr_t status = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                             H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, W.data());
    if (status < 0) {
        H5Dclose(dset);
        H5Sclose(space);
        H5Fclose(file);
        throw runtime_error("Failed to write /W");
    }

    H5Dclose(dset);
    H5Sclose(space);
    H5Fclose(file);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0]
             << " <X_filtered.h5> <W_out.h5> [k=32]\n";
        return 1;
    }

    string x_h5_path = argv[1];
    string w_h5_path = argv[2];
    hsize_t k = 32;
    if (argc >= 4) {
        k = static_cast<hsize_t>(stoi(argv[3]));
    }

    try {
        auto shape = read_matrix_shape(x_h5_path);
        hsize_t n_cells = shape.first;
        hsize_t n_genes = shape.second;

        cout << "X shape: cells=" << n_cells
             << ", genes/features=" << n_genes << "\n";
        cout << "Generating W with shape [genes=" << n_genes
             << " x k=" << k << "]\n";

        // W ~ N(0,1)
        vector<float> W(n_genes * k);
        mt19937 rng(0);
        normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : W) v = dist(rng);

        write_W_h5(w_h5_path, W, n_genes, k);
        cout << "W written to " << w_h5_path << " (dataset /W)\n";
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
