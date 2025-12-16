# Build script for row+col permutation test
# Usage: .\build_test_perm_2d_spmm.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Row+Col Permutation Test (test_perm2d)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Compiler settings
$CXX = "g++"
$CXXFLAGS = "-std=c++17 -O3 -Wall -I../include"

# Source files
$SOURCES = @("../source/test_perm_2d.cpp", "../source/permutation.cpp", "../source/disk_to_memory.cpp")
$OUTPUT = "../build/test_perm2d.exe"

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
     Write-Host "Usage: .\..\build\test_perm2d.exe [optional: x_file w_file ...]" -ForegroundColor Cyan
    Write-Host "Default: Runs test cases (d0/w0, d2/w2, d3/w3, d4/w4, d5/w5)" -ForegroundColor Yellow
    Write-Host "Example: .\..\build\test_perm2d.exe" -ForegroundColor Yellow
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}
