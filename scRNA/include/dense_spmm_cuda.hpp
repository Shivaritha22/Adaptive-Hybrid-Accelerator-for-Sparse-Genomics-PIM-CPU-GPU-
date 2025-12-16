#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>

/**
  CUDA Error Checking Macros
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(1); \
        } \
    } while(0)

/**
 Dense SpMM on GPU using cuBLAS
 
 Performs Y = X * W where:
 - X is M × K (row-major, dense)
 - W is K × N (row-major, dense)
 - Y is M × N (row-major, dense)
 
 All pointers are host pointers. The function handles device memory allocation,
 data transfer, cuBLAS SGEMM call, and result transfer back to host.
 
 @param h_X Host pointer to X matrix (M × K, row-major)
 @param h_W Host pointer to W matrix (K × N, row-major)
 @param h_Y Host pointer to Y matrix (M × N, row-major, output)
 @param M Number of rows in X and Y
 @param K Number of columns in X and rows in W
 @param N Number of columns in W and Y
 */
void dense_spmm_cuda_tile(
    const float* h_X,
    const float* h_W,
    float*       h_Y,
    int M, int K, int N);

/**
 CUDA Device Information Structure
 */
struct CudaDeviceInfo {
    std::string device_name;
    int compute_major;
    int compute_minor;
    std::string cuda_runtime_version;
    std::string cublas_version;
    int device_id;
    size_t total_memory_mb;
    bool available;
    
    CudaDeviceInfo() : compute_major(0), compute_minor(0), device_id(0), 
                       total_memory_mb(0), available(false) {}
};

/**
 Query CUDA device information at runtime
 
 @return CudaDeviceInfo structure with GPU details, or empty/invalid if CUDA unavailable
 */
CudaDeviceInfo get_cuda_device_info();

#endif // USE_CUDA

