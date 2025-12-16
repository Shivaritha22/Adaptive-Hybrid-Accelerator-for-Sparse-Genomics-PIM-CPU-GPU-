# Build script for Test 3: Predictor + Tiled SpMM (PIM OFF)
# Usage: .\build_test3.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Test 3: Predictor + Tiled SpMM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Compiler settings
$CXX = "g++"
$CXXFLAGS = "-std=c++17 -O3 -Wall -I../include"

# Source files
$SOURCES = @("../source/test3_predicted_tiled_spmm.cpp", "../source/disk_to_memory.cpp", "../source/tiler.cpp", "../source/spmm.cpp", "../source/spmm_baseline.cpp", "../source/dense_spmm.cpp")
$OUTPUT = "../build/run3.exe"

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
    Write-Host "Usage: .\..\build\run3.exe X_file.h5 W_file.h5" -ForegroundColor Cyan
    Write-Host "Example: .\..\build\run3.exe d0.h5 w0.h5" -ForegroundColor Yellow
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}














