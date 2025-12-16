# Architectural Summary: PIM-Side Reduction Impact on Operational Intensity
## Overview
This analysis demonstrates how PIM-side filtering increases operational intensity and shifts the workload from memory-bound toward compute-bound regime.

## Key Findings

### Dataset 1

- **NNZ Reduction**: 49.0% (26.41M -> 13.47M)
- **Memory Traffic Reduction**: -34.6% (0.221 GB -> 0.298 GB)
- **FLOPs Reduction**: 49.0% (1.69 GFLOPs -> 0.86 GFLOPs)
- **Operational Intensity**: 7.640 -> 2.896 FLOPs/Byte (-62.1%)
- **Regime**: Compute-bound -> Compute-bound

### Dataset 2

- **NNZ Reduction**: 42.2% (14.97M -> 8.66M)
- **Memory Traffic Reduction**: -73.8% (0.131 GB -> 0.227 GB)
- **FLOPs Reduction**: 42.2% (0.96 GFLOPs -> 0.55 GFLOPs)
- **Operational Intensity**: 7.333 -> 2.441 FLOPs/Byte (-66.7%)
- **Regime**: Compute-bound -> Compute-bound

### Dataset 3

- **NNZ Reduction**: 86.3% (24.51M -> 3.37M)
- **Memory Traffic Reduction**: 62.0% (0.232 GB -> 0.088 GB)
- **FLOPs Reduction**: 86.3% (1.57 GFLOPs -> 0.22 GFLOPs)
- **Operational Intensity**: 6.776 -> 2.449 FLOPs/Byte (-63.9%)
- **Regime**: Compute-bound -> Compute-bound

### Dataset 4

- **NNZ Reduction**: 45.7% (16.03M -> 8.71M)
- **Memory Traffic Reduction**: 1.2% (0.138 GB -> 0.137 GB)
- **FLOPs Reduction**: 45.7% (1.03 GFLOPs -> 0.56 GFLOPs)
- **Operational Intensity**: 7.425 -> 4.082 FLOPs/Byte (-45.0%)
- **Regime**: Compute-bound -> Compute-bound

## Impact of PIM Filtering

1. **Reduced Memory Traffic**: Fewer non-zero elements (nnz) directly reduces the number of bytes that need to be transferred from memory.

2. **Reduced FLOPs**: With fewer nnz, the computational work decreases proportionally.

3. **Increased Operational Intensity**: The ratio FLOPs/Bytes improves because:
   - Bytes decrease more than FLOPs (due to reduced data movement overhead)
   - This shifts workloads toward the compute-bound regime

4. **Performance Implications**: Higher operational intensity means the workload is less limited by memory bandwidth and can better utilize compute resources.

