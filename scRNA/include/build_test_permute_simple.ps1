# Build script for simple permutation test
# Usage: .\build_test_permute_simple.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Simple Permutation Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Compiler settings
$CXX = "g++"
$CXXFLAGS = "-std=c++17 -O3 -Wall -I../include"

# Source files
$SOURCES = @("../source/test_permute_simple.cpp", "../source/permutation.cpp")
$OUTPUT = "../build/test_permute_simple.exe"

Write-Host "Compiling..." -ForegroundColor Yellow

$buildCmd = "$CXX $CXXFLAGS $($SOURCES -join ' ') -o $OUTPUT"

Write-Host $buildCmd -ForegroundColor Gray
Write-Host ""

Invoke-Expression $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\..\build\test_permute_simple.exe" -ForegroundColor Cyan
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

