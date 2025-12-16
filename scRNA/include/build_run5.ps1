# Build script for CUDA Tiled SpMM Test (run5)
# Usage: .\build_run5.ps1
#
# Note: This script builds with CUDA support if CUDA is available.
# To build without CUDA, remove -DUSE_CUDA and CUDA-related flags.

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building CUDA Tiled SpMM Test (run5)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Compiler settings
$CXX = "g++"
$CXXFLAGS = "-std=c++17 -O3 -Wall -fopenmp -I../include"

# Check if CUDA is available (you may need to adjust these paths)
$CUDA_AVAILABLE = $false
$CUDA_PATH = $env:CUDA_PATH
if ($CUDA_PATH) {
    $CUDA_AVAILABLE = $true
    Write-Host "CUDA detected at: $CUDA_PATH" -ForegroundColor Green
} else {
    Write-Host "CUDA not found in environment. Building without CUDA support." -ForegroundColor Yellow
    Write-Host "Set CUDA_PATH environment variable to enable CUDA." -ForegroundColor Yellow
}

# Source files
$SOURCES = @(
    "../source/test_run5_cuda_tiled_spmm.cpp",
    "../source/permutation.cpp",
    "../source/disk_to_memory.cpp",
    "../source/spmm_baseline.cpp",
    "../source/tiler.cpp",
    "../source/tile_spmm.cpp"
)

# Add CUDA source if available
if ($CUDA_AVAILABLE) {
    $CXXFLAGS += " -DUSE_CUDA"
    $SOURCES += "../source/dense_spmm_cuda.cu"
    Write-Host "CUDA: ENABLED" -ForegroundColor Green
} else {
    Write-Host "CUDA: DISABLED (CPU fallback will be used)" -ForegroundColor Yellow
}

$OUTPUT = "../build/run5.exe"

Write-Host ""
Write-Host "Compiling..." -ForegroundColor Yellow

# Get HDF5 flags from pkg-config
$hdf5Flags = pkg-config --cflags --libs hdf5

# CUDA flags (if available)
$cudaFlags = ""
$cudaObjFile = ""
if ($CUDA_AVAILABLE) {
    $cudaInclude = "$CUDA_PATH/include"
    # Try to find the correct lib directory (Windows: lib/x64, Linux: lib64)
    $cudaLib = ""
    if (Test-Path "$CUDA_PATH/lib/x64") {
        $cudaLib = "$CUDA_PATH/lib/x64"
    } elseif (Test-Path "$CUDA_PATH/lib64") {
        $cudaLib = "$CUDA_PATH/lib64"
    } elseif (Test-Path "$CUDA_PATH/lib") {
        $cudaLib = "$CUDA_PATH/lib"
    } else {
        Write-Host "Warning: Could not find CUDA lib directory" -ForegroundColor Yellow
        $cudaLib = "$CUDA_PATH/lib"
    }
    $cudaFlags = "-I$cudaInclude -L$cudaLib -lcudart -lcublas"
    
    # Compile CUDA file separately with nvcc
    $nvcc = "$CUDA_PATH/bin/nvcc"
    if (Test-Path $nvcc) {
        $cudaObjFile = "../build/dense_spmm_cuda.o"
        Write-Host "Compiling CUDA file..." -ForegroundColor Yellow
        # nvcc compilation: -c for object file, -std=c++17, -O3, include paths, -x cu for CUDA source
        $nvccCmd = "$nvcc -c -std=c++17 -O3 -I../include -I$cudaInclude -x cu ../source/dense_spmm_cuda.cu -o $cudaObjFile"
        Write-Host $nvccCmd -ForegroundColor Gray
        Invoke-Expression $nvccCmd
        if ($LASTEXITCODE -ne 0) {
            Write-Host "CUDA compilation failed!" -ForegroundColor Red
            exit $LASTEXITCODE
        }
        # Remove CUDA source from sources list and add object file
        $SOURCES = $SOURCES | Where-Object { $_ -ne "../source/dense_spmm_cuda.cu" }
        $SOURCES += $cudaObjFile
    } else {
        Write-Host "nvcc not found at $nvcc" -ForegroundColor Red
        Write-Host "Cannot compile CUDA code without nvcc. Exiting." -ForegroundColor Red
        exit 1
    }
}

# Build command
$buildCmd = "$CXX $CXXFLAGS $($SOURCES -join ' ') -o $OUTPUT $hdf5Flags -lhdf5_cpp $cudaFlags"

Write-Host $buildCmd -ForegroundColor Gray
Write-Host ""

Invoke-Expression $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\..\build\run5.exe <X_file.h5> <W_file.h5>" -ForegroundColor Cyan
    Write-Host "Example: .\..\build\run5.exe d5.h5 w5.h5" -ForegroundColor Yellow
    if ($CUDA_AVAILABLE) {
        Write-Host ""
        Write-Host "Note: CUDA support is enabled. GPU will be used for dense tiles." -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Note: CUDA support is disabled. CPU fallback will be used for dense tiles." -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "Build failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  - If CUDA errors: Make sure CUDA is installed and CUDA_PATH is set" -ForegroundColor Yellow
    Write-Host "  - If HDF5 errors: Make sure HDF5 is installed and pkg-config can find it" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

