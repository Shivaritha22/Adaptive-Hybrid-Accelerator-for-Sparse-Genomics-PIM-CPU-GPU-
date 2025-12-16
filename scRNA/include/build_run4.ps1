# Build script for Tiled Predictor SpMM Test (run4)
# Usage: .\build_run4.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Tiled Predictor SpMM Test (run4)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Compiler settings
$CXX = "g++"
$CXXFLAGS = "-std=c++17 -O3 -Wall -fopenmp -I../include"

# Source files
$SOURCES = @("../source/test_tiled_predictor_spmm.cpp", "../source/permutation.cpp", "../source/disk_to_memory.cpp", "../source/spmm_baseline.cpp", "../source/tiler.cpp", "../source/tile_spmm.cpp")
$OUTPUT = "../build/run4.exe"

Write-Host "Compiling..." -ForegroundColor Yellow

# Get HDF5 flags from pkg-config
$hdf5Flags = pkg-config --cflags --libs hdf5

# Build command
$buildCmd = "$CXX $CXXFLAGS $($SOURCES -join ' ') -o $OUTPUT $hdf5Flags -lhdf5_cpp"

Write-Host $buildCmd -ForegroundColor Gray
Write-Host ""

Invoke-Expression $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\..\build\run4.exe <X_file.h5> <W_file.h5>" -ForegroundColor Cyan
    Write-Host "Example: .\..\build\run4.exe d5.h5 w5.h5" -ForegroundColor Yellow
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

