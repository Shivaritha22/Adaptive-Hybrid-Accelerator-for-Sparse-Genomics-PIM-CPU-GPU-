#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 64   // Rows of X
#define K 128  // Columns of X / Rows of W
#define N 96   // Columns of W
#define WARP_SIZE 32

// CPU reference implementation
void cpu_matmul(float *X, float *W, float *Y, int rows, int cols, int inner) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < inner; k++) {
                sum += X[i * inner + k] * W[k * cols + j];
            }
            Y[i * cols + j] = sum;
        }
    }
}

// Warp-level GPU kernel
// Each block is one warp (32 threads)
// Each warp computes one output element Y[row, col]
__global__ void warp_matmul_kernel(float *X, float *W, float *Y, int rows, int cols, int inner) {
    // Calculate which output element this warp is responsible for
    int row = blockIdx.y;
    int col = blockIdx.x;
    
    // Each thread in the warp handles different k positions
    int lane_id = threadIdx.x;
    
    // Accumulate partial dot product
    float sum = 0.0f;
    for (int k = lane_id; k < inner; k += WARP_SIZE) {
        sum += X[row * inner + k] * W[k * cols + col];
    }
    
    // Warp reduction using __shfl_down_sync
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float val = __shfl_down_sync(0xFFFFFFFF, sum, offset);
        sum += val;
    }
    
    // Lane 0 writes the final result
    if (lane_id == 0) {
        Y[row * cols + col] = sum;
    }
}

// Compute max absolute difference between two matrices
float max_abs_error(float *Y_ref, float *Y_gpu, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = fabsf(Y_ref[i] - Y_gpu[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

// Write matrix to file
void write_matrix_to_file(float *matrix, int rows, int cols, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Write dimensions
    fprintf(fp, "%d %d\n", rows, cols);
    
    // Write matrix values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.8e", matrix[i * cols + j]);
            if (j < cols - 1) {
                fprintf(fp, " ");
            }
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    printf("Matrix written to %s\n", filename);
}

int main() {
    // Allocate host memory
    size_t size_X = M * K * sizeof(float);
    size_t size_W = K * N * sizeof(float);
    size_t size_Y = M * N * sizeof(float);
    
    float *h_X = (float *)malloc(size_X);
    float *h_W = (float *)malloc(size_W);
    float *h_Y_ref = (float *)malloc(size_Y);
    float *h_Y_gpu = (float *)malloc(size_Y);
    
    // Initialize test matrices with fixed values
    // Fill X with pattern: X[i][j] = (i * K + j) * 0.01f
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_X[i * K + j] = (i * K + j) * 0.01f;
        }
    }
    
    // Fill W with pattern: W[i][j] = (i * N + j) * 0.02f
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_W[i * N + j] = (i * N + j) * 0.02f;
        }
    }
    
    // Compute reference on CPU
    printf("Computing reference on CPU...\n");
    cpu_matmul(h_X, h_W, h_Y_ref, M, K, N);
    
    // Allocate device memory
    float *d_X, *d_W, *d_Y;
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_W, size_W);
    cudaMalloc(&d_Y, size_Y);
    
    // Copy data to device
    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
    
    // Launch kernel
    // Grid: (N, M) - one block per output element
    // Block: (32, 1, 1) - one warp per block
    dim3 gridDim(N, M);
    dim3 blockDim(WARP_SIZE, 1, 1);
    
    printf("Launching warp-level GPU kernel...\n");
    warp_matmul_kernel<<<gridDim, blockDim>>>(d_X, d_W, d_Y, M, K, N);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Check for kernel execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_Y_gpu, d_Y, size_Y, cudaMemcpyDeviceToHost);
    
    // Compare results
    float max_err = max_abs_error(h_Y_ref, h_Y_gpu, M * N);
    printf("Max abs error: %.6e\n", max_err);
    
    // Write matrices to files
    write_matrix_to_file(h_Y_ref, M, N, "Y_ref.txt");
    write_matrix_to_file(h_Y_gpu, M, N, "Y_warp.txt");
    
    // Cleanup
    free(h_X);
    free(h_W);
    free(h_Y_ref);
    free(h_Y_gpu);
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);
    
    return 0;
}

