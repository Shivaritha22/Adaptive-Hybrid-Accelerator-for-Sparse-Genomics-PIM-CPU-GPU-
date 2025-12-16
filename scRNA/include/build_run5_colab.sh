#!/bin/bash
# Build script for CUDA Tiled SpMM Test (run5) - Google Colab / Linux
# Usage: bash build_run5_colab.sh

echo "========================================"
echo "Building CUDA Tiled SpMM Test (run5)"
echo "========================================"
echo ""

# Compiler settings
CXX="g++"
CXXFLAGS="-std=c++17 -O3 -Wall -fopenmp -I../include"

# Check if CUDA is available
CUDA_AVAILABLE=false
if [ -n "$CUDA_PATH" ]; then
    CUDA_AVAILABLE=true
    echo "CUDA detected at: $CUDA_PATH"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
    CUDA_AVAILABLE=true
    echo "CUDA detected at: $CUDA_PATH"
else
    echo "CUDA not found. Building without CUDA support."
    echo "Note: Google Colab usually has CUDA at /usr/local/cuda"
fi

# Source files
SOURCES=(
    "../source/test_run5_cuda_tiled_spmm.cpp"
    "../source/permutation.cpp"
    "../source/disk_to_memory.cpp"
    "../source/spmm_baseline.cpp"
    "../source/tiler.cpp"
    "../source/tile_spmm.cpp"
)

# Add CUDA source if available
if [ "$CUDA_AVAILABLE" = true ]; then
    CXXFLAGS="$CXXFLAGS -DUSE_CUDA"
    echo "CUDA: ENABLED"
else
    echo "CUDA: DISABLED (CPU fallback will be used)"
fi

OUTPUT="../build/run5"

echo ""
echo "Compiling..."

# Get HDF5 flags from pkg-config
HDF5_FLAGS=$(pkg-config --cflags --libs hdf5 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Warning: pkg-config not found or HDF5 not configured. Trying default paths..."
    HDF5_FLAGS="-I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lhdf5_cpp"
fi

# CUDA flags (if available)
CUDA_FLAGS=""
if [ "$CUDA_AVAILABLE" = true ]; then
    CUDA_INCLUDE="$CUDA_PATH/include"
    CUDA_LIB="$CUDA_PATH/lib64"
    
    # Compile CUDA file separately with nvcc
    NVCC="$CUDA_PATH/bin/nvcc"
    if [ -f "$NVCC" ]; then
        CUDA_OBJ_FILE="../build/dense_spmm_cuda.o"
        echo "Compiling CUDA file..."
        NVCC_CMD="$NVCC -c -std=c++17 -O3 -I../include -I$CUDA_INCLUDE -x cu ../source/dense_spmm_cuda.cu -o $CUDA_OBJ_FILE"
        echo "$NVCC_CMD"
        $NVCC_CMD
        if [ $? -ne 0 ]; then
            echo "CUDA compilation failed!"
            exit 1
        fi
        # Remove CUDA source from sources list and add object file
        # Note: dense_spmm_cuda.cu was never added to SOURCES, so we just add the object file
        SOURCES+=("$CUDA_OBJ_FILE")
    else
        echo "nvcc not found at $NVCC"
        exit 1
    fi
    
    CUDA_FLAGS="-I$CUDA_INCLUDE -L$CUDA_LIB -lcudart -lcublas"
fi

# Create build directory if it doesn't exist
mkdir -p ../build

# Build command
BUILD_CMD="$CXX $CXXFLAGS ${SOURCES[@]} -o $OUTPUT $HDF5_FLAGS $CUDA_FLAGS"

echo "$BUILD_CMD"
echo ""

$BUILD_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""
    echo "Usage: ./build/run5 <X_file.h5> <W_file.h5>"
    echo "Example: ./build/run5 d5.h5 w5.h5"
    if [ "$CUDA_AVAILABLE" = true ]; then
        echo ""
        echo "Note: CUDA support is enabled. GPU will be used for dense tiles."
    else
        echo ""
        echo "Note: CUDA support is disabled. CPU fallback will be used for dense tiles."
    fi
else
    echo ""
    echo "Build failed!"
    echo ""
    echo "Troubleshooting:"
    echo "  - If CUDA errors: Make sure CUDA is installed (usually at /usr/local/cuda in Colab)"
    echo "  - If HDF5 errors: Install HDF5: sudo apt-get install libhdf5-dev"
    exit 1
fi

