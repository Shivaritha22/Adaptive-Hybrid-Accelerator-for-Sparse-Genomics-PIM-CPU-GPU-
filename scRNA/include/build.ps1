# Build script for sparse matrix multiplication baseline
# Usage: .\build.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Sparse Matrix Multiplication" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Compiler settings
$CXX = "g++"
$CXXFLAGS = "-std=c++17 -O3 -Wall -fopenmp -I../include"

# Source files (from source/ directory)
$SOURCES = @("../source/main.cpp", "../source/disk_to_memory.cpp", "../source/spmm_baseline.cpp")
$OUTPUT = "../build/run4.exe"
$ALT_OUTPUT = "../build/run0.exe"

Write-Host "Compiling..." -ForegroundColor Yellow

# Get HDF5 flags from pkg-config
# Note: Ensure pkg-config is in your PATH
$hdf5Flags = pkg-config --cflags --libs hdf5

# Build command (add -lhdf5_cpp for C++ API)
$buildCmd = "$CXX $CXXFLAGS $($SOURCES -join ' ') -o $OUTPUT $hdf5Flags -lhdf5_cpp"

Write-Host $buildCmd -ForegroundColor Gray
Write-Host ""

Invoke-Expression $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\..\build\run4.exe X_file.h5 W_file.h5" -ForegroundColor Cyan
    Write-Host "Example: .\..\build\run4.exe d0.h5 w0.h5" -ForegroundColor Yellow
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}
