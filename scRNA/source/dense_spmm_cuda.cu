#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <string>

#include "../include/dense_spmm_cuda.hpp"


// Error-checking macros
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t st__ = (call);                                        \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",                    \
                    static_cast<int>(st__), __FILE__, __LINE__);             \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)
#endif


// Dense SpMM tile on CUDA using cuBLAS
// X: row-major  M x K
// W: row-major  K x N
// Y: row-major  M x N


void dense_spmm_cuda_tile(
    const float* h_X,
    const float* h_W,  
    float*       h_Y, 
    int M, int K, int N)
{
    // device memory
    float *d_X, *d_W, *d_Y;
    size_t size_X = static_cast<size_t>(M) * K * sizeof(float);
    size_t size_W = static_cast<size_t>(K) * N * sizeof(float);
    size_t size_Y = static_cast<size_t>(M) * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_X, size_X));
    CUDA_CHECK(cudaMalloc((void**)&d_W, size_W));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, size_Y));

    CUDA_CHECK(cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    /*
       Y_row = X_row (M x K) * W_row (K x N)     [row-major].

      Interpret row-major as transpose of column-major:

        X_row (M x K, row-major)  <=>  X^T (K x M, col-major)
        W_row (K x N, row-major)  <=>  W^T (N x K, col-major)
        Y_row (M x N, row-major)  <=>  Y^T (N x M, col-major)

      Then, in cuBLAS (column-major):

        Y^T = W^T * X^T

      Call SGEMM with:
        A = W^T (N x K)
        B = X^T (K x M)
        C = Y^T (N x M)

      Memory of Y^T (N x M, col-major) is exactly Y_row (M x N, row-major),
      copy d_Y straight back into h_Y with no extra transpose.
    */

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,  // A: W^T (N x K)
        CUBLAS_OP_N,  // B: X^T (K x M)
        N,            // m = rows of A = N
        M,            // n = cols of B = M
        K,            // k
        &alpha,
        d_W,          // A pointer (W_row as W^T)
        N,            // lda = rows of A = N
        d_X,          // B pointer (X_row as X^T)
        K,            // ldb = rows of B = K
        &beta,
        d_Y,          // C pointer (Y_row as Y^T)
        N));          // ldc = rows of C = N

    // Copy result back; already M x N row-major
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, size_Y, cudaMemcpyDeviceToHost));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_Y));
}


CudaDeviceInfo get_cuda_device_info() {
    CudaDeviceInfo info;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        info.available = false;
        return info;
    }

    info.available = true;

    int current_device = 0;
    CUDA_CHECK(cudaGetDevice(&current_device));
    info.device_id = current_device;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, current_device));

    info.device_name      = std::string(prop.name);
    info.compute_major    = prop.major;
    info.compute_minor    = prop.minor;
    info.total_memory_mb  = prop.totalGlobalMem / (1024 * 1024);

    int runtime_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    int rt_major = runtime_version / 1000;
    int rt_minor = (runtime_version % 1000) / 10;
    {
        std::stringstream ss;
        ss << rt_major << "." << rt_minor;
        info.cuda_runtime_version = ss.str();
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    int cublas_version = 0;
    CUBLAS_CHECK(cublasGetVersion(handle, &cublas_version));
    CUBLAS_CHECK(cublasDestroy(handle));

    int cb_major = cublas_version / 1000;
    int cb_minor = (cublas_version % 1000) / 100;
    int cb_patch = (cublas_version % 100) / 10;
    int cb_build = cublas_version % 10;
    {
        std::stringstream ss;
        ss << cb_major << "." << cb_minor
           << "." << cb_patch << "." << cb_build;
        info.cublas_version = ss.str();
    }

    return info;
}

#endif 
