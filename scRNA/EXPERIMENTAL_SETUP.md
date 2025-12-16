# Experimental Setup

## Hardware Configuration

### GPU Platform
- **Platform**: Google Colab
- **GPU Model**: NVIDIA L4
- **GPU Type**: L4 (as specified in Colab configuration)
- **Machine Shape**: High-memory (hm)

### CPU Platform
- **CPU**: Automatically assigned by Google Colab when L4 GPU is selected
- **CPU Type**: Typically Intel Xeon processors (exact model varies by Colab instance allocation)
- **Parallel Threads**: 16 OpenMP threads (as observed in execution logs)
- **Role**: Processes sparse tiles (<5% density) in the hybrid CPU/GPU execution model
- **Note**: Google Colab dynamically allocates CPU resources; the specific CPU model is not guaranteed to be consistent across sessions

### Compute Environment
- **CUDA Runtime**: Available via `/usr/local/cuda`
- **cuBLAS Library**: Integrated with CUDA installation
- **Memory**: GPU memory managed by CUDA runtime
- **Hybrid Execution**: Dense tiles (≥5% density) processed on GPU; sparse tiles (<5% density) processed on CPU

## Software Environment

### Compiler and Build Tools
- **C++ Compiler**: `g++` with C++17 standard (`-std=c++17`)
- **CUDA Compiler**: `nvcc` (NVIDIA CUDA Compiler)
- **Optimization Level**: `-O3` (maximum optimization)
  - **What it means**: `-O3` enables the highest level of compiler optimizations in GCC. The compiler performs aggressive code transformations to maximize execution speed, including:
    - **Loop optimizations**: Loop unrolling, loop vectorization (SIMD), loop fusion, and loop interchange
    - **Function inlining**: Expanding function calls inline to eliminate call overhead
    - **Dead code elimination**: Removing unreachable or unused code
    - **Constant propagation**: Replacing variables with their constant values when known
    - **Instruction scheduling**: Reordering instructions to better utilize CPU pipelines
    - **Register allocation**: Optimizing register usage to minimize memory accesses
    - **Inter-procedural optimizations**: Optimizations across function boundaries
  - **Why used**: Critical for performance in compute-intensive operations like sparse matrix multiplication, where even small improvements can significantly impact overall execution time
  - **Trade-offs**: Increases compilation time and may make debugging more difficult (optimized code may not match source code line-by-line), but essential for production performance benchmarks
  - **Comparison**: 
    - `-O0`: No optimization (fastest compilation, slowest execution, best for debugging)
    - `-O1`: Basic optimizations (balance between compile time and performance)
    - `-O2**: Standard optimizations (default for most production code)
    - `-O3`: Maximum optimization (best performance, used here for benchmarking)
- **Parallel Processing**: OpenMP enabled (`-fopenmp`)

### Libraries and Dependencies
- **HDF5**: Version 1.10.7+ (for reading/writing HDF5 format data files)
  - Installation: `libhdf5-dev` package
  - Libraries: `-lhdf5 -lhdf5_cpp`
- **CUDA Libraries**:
  - `libcudart` (CUDA Runtime)
  - `libcublas` (cuBLAS for dense matrix operations)
- **Build Configuration**: `-DUSE_CUDA` preprocessor flag enabled

### Operating System
- **OS**: Ubuntu 22.04 (Jammy) on Google Colab
- **Package Manager**: `apt-get`

## Dataset Configuration

### Data Format
- **Input Format**: HDF5 files (10x Genomics format)
- **Matrix X**: Sparse matrix stored in CSC (Compressed Sparse Column) format, converted to CSR (Compressed Sparse Row) for computation
- **Matrix W**: Dense matrix (weight matrix)
- **Output Format**: HDF5 files

### Dataset Characteristics
The experiments were conducted on multiple single-cell RNA sequencing datasets with varying dimensions:

| Dataset | X Dimensions | W Dimensions | Non-zeros (nnz) | Density |
|---------|--------------|--------------|-----------------|---------|
| d2.h5 | 33,696 × 9,263 | 9,263 × 32 | 26,407,826 | ~0.084 |
| d3.h5 | 38,606 × 6,704 | 6,704 × 32 | 14,971,913 | ~0.058 |
| d4.h5 | 134,920 × 2,711 | 2,711 × 32 | 24,511,186 | ~0.067 |
| d5.h5 | 36,706 × 2,979 | 2,979 × 32 | 16,028,697 | ~0.147 |

**Note**: Matrix X represents gene expression data (genes × cells), and matrix W represents learned weights (features × output dimensions).

## Algorithm Configuration

The algorithm parameters are defined in the `config/` folder as compile-time constants. These parameters directly control the tiling strategy, parallel execution, and hybrid CPU/GPU workload distribution.

### Hardware Configuration (`config/hw_config.h`)

This file defines fixed hardware parameters that are compile-time constants. These values represent the physical characteristics of the target hardware and must be changed by editing the file and recompiling.

**Key Parameters:**
- **Tile Dimensions**: 
  - `TILE_ROWS = 64`: Number of rows per tile
  - `TILE_COLS = 64`: Number of columns per tile
  - All matrices are partitioned into 64×64 tiles for processing

- **Thread Configuration**:
  - `NUM_THREADS = 8`: Number of threads for parallel execution (base configuration)
  - Note: Actual execution used 16 OpenMP threads (as observed in logs), which may be set by OpenMP runtime based on available CPU cores

- **Dense Tile Threshold**:
  - `DENSE_TILE_THRESHOLD = 0.05`: Tiles with density ≥ 5% are classified as dense
  - **Dense tiles** (density ≥ 5%): Routed to GPU for processing using cuBLAS SGEMM
  - **Sparse tiles** (density < 5%): Routed to CPU for processing using optimized sparse kernels
  - This threshold determines the hybrid CPU/GPU workload distribution

**Configuration Impact:**
- Larger tile sizes (64×64) help amortize GPU memory transfer overhead
- The 5% density threshold balances GPU utilization with overhead costs
- These parameters directly affect performance metrics and must be reported for reproducibility

### PIM Configuration (`config/pim_defaults.h` and `include/pim_config.h`)

These files define Processing-In-Memory (PIM) configuration parameters, including:
- Filter modes (None, ValueThreshold)
- Quantization modes (None, Int8PerRow, Int8Global)
- Default filtering and quantization parameters

**Note**: For the experiments reported here, PIM features were set to default (no filtering, no quantization), focusing on the hybrid CPU/GPU tiling approach.

## Build Configuration

### Compilation Steps
1. **CUDA Object Compilation**:
   ```bash
   nvcc -c -std=c++17 -O3 -DUSE_CUDA -Iinclude -I/usr/local/cuda/include \
        -x cu source/dense_spmm_cuda.cu -o build/dense_spmm_cuda.o
   ```

2. **Main Executable Build**:
   ```bash
   g++ -std=c++17 -O3 -Wall -fopenmp -DUSE_CUDA -Iinclude \
       source/test_run5_cuda_tiled_spmm.cpp \
       source/permutation.cpp \
       source/disk_to_memory.cpp \
       source/spmm_baseline.cpp \
       source/tiler.cpp \
       source/tile_spmm.cpp \
       build/dense_spmm_cuda.o \
       -o build/run5 \
       -I/usr/include/hdf5/serial \
       -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
       -lhdf5 -lhdf5_cpp \
       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas
   ```

### Key Compilation Flags
- `-std=c++17`: C++17 standard
- `-O3`: Maximum optimization
- `-fopenmp`: OpenMP support for parallel processing
- `-DUSE_CUDA`: Enable CUDA support
- `-Wall`: Enable all compiler warnings

## Experimental Methodology

### Computation Pipeline
1. **Data Loading**: Load sparse matrix X and dense matrix W from HDF5 files
2. **Format Conversion**: Convert X from CSC to CSR format
3. **Tiling**: Partition matrices into 64×64 tiles
4. **Tile Classification**: Classify tiles as dense (≥5% density) or sparse (<5% density)
5. **Hybrid Execution**:
   - Dense tiles: Processed on GPU using cuBLAS SGEMM
   - Sparse tiles: Processed on CPU using optimized sparse kernels
6. **Result Aggregation**: Combine results from all tiles
7. **Output**: Save result matrix Y to HDF5 file



### Environment Setup
1. Google Colab notebook with GPU runtime (L4)
2. Install dependencies: `apt-get install -y libhdf5-dev pkg-config`
3. Mount Google Drive (if datasets stored there)
4. Build using provided compilation commands
5. Execute with dataset files: `./build/run5 d2.h5 w2.h5`

### Hardware Information Verification
To check CPU information in Google Colab (if needed for detailed reporting):
```bash
# Check CPU model and specifications
lscpu | grep "Model name"
cat /proc/cpuinfo | grep "model name" | head -1

# Check number of CPU cores/threads
nproc
lscpu | grep "^CPU(s):"

# Check OpenMP thread count (as used in execution)
# This is typically set automatically by OpenMP based on available cores
```

**Note**: Google Colab dynamically allocates CPU resources, so the exact CPU model may vary between sessions. The experiments consistently used 16 OpenMP threads as observed in the execution logs.

### File Structure
- `source/`: C++ source files
- `include/`: Header files
- `config/`: Configuration files (hardware parameters)
- `dataset/X/`: Input sparse matrices
- `dataset/W/`: Input dense weight matrices
- `dataset/Y/`: Output result matrices
- `logs/`: Performance logs and timing information
- `build/`: Compiled executables


