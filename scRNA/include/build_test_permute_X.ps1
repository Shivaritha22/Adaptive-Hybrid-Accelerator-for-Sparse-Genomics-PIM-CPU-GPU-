# Build script for X permutation test
# Usage: .\build_test_permute_X.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building X Permutation Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Compiler settings
$CXX = "g++"
$CXXFLAGS = "-std=c++17 -O3 -Wall -I../include"

# Source files
$SOURCES = @("../source/test_permute_X_datasets.cpp", "../source/permutation.cpp", "../source/disk_to_memory.cpp")
$OUTPUT = "../build/test_permute_X.exe"

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
    Write-Host "Usage: .\..\build\test_permute_X.exe [d0.h5 d2.h5 ...]" -ForegroundColor Cyan
    Write-Host "Example: .\..\build\test_permute_X.exe" -ForegroundColor Yellow
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

