# Sanity Check for Numerical Error Validation

## Purpose
The sanity check is performed to validate computational results and detect numerical errors when comparing outputs from different implementations (e.g., CUDA vs reference CPU implementations).

## Floating Point Arithmetic and IEEE 754
According to the IEEE 754 standard, floating-point arithmetic operations are not associative. This means that the order of operations can affect the final result. For example:

- `(a + b) + c` may not equal `a + (b + c)` due to rounding errors
- Similarly, `a + b + c` computed in different orders can yield different results

This non-associativity becomes particularly significant in:
- **Parallel computations** (CUDA/GPU): Operations may be executed in different orders compared to sequential CPU implementations
- **Accumulation operations**: Summing many values can accumulate rounding errors differently
- **Matrix operations**: Different parallelization strategies can lead to different computation orders

## Tolerance Values
To account for these expected numerical differences while still detecting genuine errors, tolerance values are set:

- **Relative tolerance (`tolerance_min`)**: `1e-3` (0.001 or 0.1%)
  - Accounts for proportional differences in larger values
- **Absolute tolerance (`tolerance_max`)**: `1e-5` (0.00001)
  - Accounts for small absolute differences, especially important for values near zero

These tolerances allow the sanity check to distinguish between:
- **Acceptable numerical variations** due to floating-point arithmetic properties
- **Actual computational errors** that exceed reasonable tolerance bounds

## Fact Check Summary
✅ **Correct**: Floating-point addition is not associative under IEEE 754  
✅ **Correct**: Tolerance values help account for expected numerical differences  
✅ **Correct**: This is especially relevant when comparing parallel (CUDA) vs sequential implementations  
⚠️ **Note**: The tolerance values are empirically chosen based on the expected magnitude of floating-point errors in the specific computations being compared

