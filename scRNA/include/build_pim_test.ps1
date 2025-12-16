# Build script for PIM test with real data
# Tests PIM emulator on actual H5 datasets

Write-Host "Building PIM test with real data..." -ForegroundColor Cyan

$sources = @("../source/pim_test_real_data.cpp", "../source/disk_to_memory.cpp", "../source/pim_filter.cpp", "../source/pim_tuner.cpp", "../source/pim_emu.cpp", "../source/spmm_baseline.cpp")
$output = "../build/pim_filter.exe"

# Get HDF5 flags
$hdf5_flags = (pkg-config --cflags --libs hdf5).Split()

$cmd = "g++ -std=c++17 -O3 -Wall -I../include $($sources -join ' ') -o $output $($hdf5_flags -join ' ') -lhdf5_cpp"

Write-Host "Command: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful: $output" -ForegroundColor Green
    Write-Host ""
    Write-Host "Run with: .\..\build\pim_filter.exe d0.h5 w0.h5" -ForegroundColor Yellow
} else {
    Write-Host "Build failed" -ForegroundColor Red
    exit 1
}
