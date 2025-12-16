# Warp-Level Matrix Multiplication

A standalone CUDA program that multiplies two dense matrices using a warp-level GPU kernel and compares the result with a CPU reference implementation.

## Features

- CPU reference implementation using triple for-loops
- GPU implementation using warp-level parallelism:
  - Each block is one warp (32 threads)
  - Each warp computes one output element
  - Each thread accumulates partial dot products
  - Warp reduction using `__shfl_down_sync`
  - Lane 0 writes the final result

## Matrix Dimensions

- X: M × K (default: 64 × 128)
- W: K × N (default: 128 × 96)
- Y: M × N (default: 64 × 96)

## Building

```bash
make
```

Or manually:
```bash
nvcc -O3 -arch=sm_75 -std=c++11 -o warp_matmul warp_matmul.cu
```

Note: Adjust `-arch=sm_75` to match your GPU's compute capability if needed.

## Running

```bash
./warp_matmul
```

On Windows:
```bash
warp_matmul.exe
```

## Output

The program will print:
- Status messages during computation
- `Max abs error: <value>` - the maximum absolute difference between CPU and GPU results

A very small error (e.g., < 1e-5) indicates the GPU kernel matches the CPU reference.

