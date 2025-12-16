# Results

## Dataset Characterization

We evaluated our hybrid CPU/GPU tiling pipeline on single-cell RNA sequencing datasets with varying sparsity patterns. Table 1 presents the characteristics of the first dataset (d2.h5) used in our experiments.

**Table 1: Dataset Characteristics (d2.h5)**

| Property | Value |
|----------|-------|
| Matrix X Dimensions | 33,696 × 9,263 |
| Matrix W Dimensions | 9,263 × 32 |
| Non-zeros (nnz) | 26,407,826 |
| Density | 0.084 (8.4%) |
| Matrix Format | Sparse (CSR) × Dense |

The dataset represents gene expression data with 33,696 genes across 9,263 cells, exhibiting a low density of 8.4% non-zero entries, which is characteristic of single-cell RNA sequencing data where most genes are not expressed in most cells.

**Take-away:** The d2 dataset demonstrates typical scRNA-seq sparsity patterns, making it an ideal candidate for evaluating sparse matrix multiplication optimizations.

## Processing-In-Memory (PIM) Data Reduction

Prior to computation, the dataset underwent PIM-based filtering to reduce computational workload. Table 2 summarizes the PIM filtering results for dataset d2.

**Table 2: PIM Filtering Results (d2.h5)**

| Metric | Original | Filtered | Reduction |
|--------|----------|----------|-----------|
| Genes (rows) | 33,696 | 3,370 | 90.0% |
| Cells (cols) | 9,263 | 9,263 | 0% |
| Non-zeros | 26,407,826 | 13,469,787 | 49.09% |
| Rare genes retained | - | 3,370 | - |
| Noise genes filtered | - | 2,451 | - |
| Genes with zero variance | - | 9,184 | - |
| Data reduction ratio | - | 0.5091 | 50.91% of original |

The PIM filtering process reduced the number of genes from 33,696 to 3,370 by eliminating noise genes and genes with zero variance, while retaining rare genes that may contain biologically relevant information. This resulted in a 49.09% reduction in non-zero entries, effectively halving the computational workload while preserving the essential structure of the data.

**Take-away:** PIM filtering achieved a 49% reduction in non-zero entries by removing noise and zero-variance genes, significantly reducing computational requirements while maintaining biological relevance.

## Performance Characterization

We evaluated the performance of our hybrid CPU/GPU tiling pipeline on the PIM-filtered d2 dataset. Table 3 compares the baseline performance against the tiled implementation with permutation and predictor optimizations.

**Table 3: Performance Results for d2.h5 (Character Study)**

| Implementation | Compute Time (ms) | Performance (GFLOP/s) | Bandwidth (GB/s) | Speedup |
|----------------|-------------------|----------------------|------------------|---------|
| Baseline (original) | 91.558 | 18.46 | 2.42 | 1.00× |
| Tiled + Perm + Predictor (PIM-filtered) | 2,069.807 | 0.42 | 0.14 | 0.04× |

The baseline implementation on the original unfiltered dataset achieved 18.46 GFLOP/s with a compute time of 91.558 ms. However, when processing the PIM-filtered dataset through the tiled pipeline with permutation and predictor optimizations, the compute time increased to 2,069.807 ms, resulting in 0.42 GFLOP/s. This performance degradation can be attributed to the overhead introduced by the tiling strategy, permutation operations, and predictor-based tile classification on the reduced dataset, where the benefits of GPU acceleration for dense tiles may not fully compensate for the added overhead.

**Take-away:** While PIM filtering reduces data size by 49%, the tiled pipeline introduces significant overhead on the filtered dataset, suggesting that the tiling strategy needs optimization for smaller, filtered datasets.

## Operational Intensity and Roofline Analysis

The operational intensity (OI) of the SpMM operation for dataset d2 was calculated as the ratio of floating-point operations to bytes transferred. For the baseline implementation, the OI was approximately 7.64 FLOP/byte (1,690,100,864 FLOPs / 221,209,236 bytes). This places the computation in the compute-bound regime for modern GPUs, where performance is limited by computational throughput rather than memory bandwidth.

The roofline model analysis reveals that the baseline implementation achieves 18.46 GFLOP/s, which is well below the theoretical peak performance of the NVIDIA L4 GPU. The tiled implementation with PIM-filtered data achieves only 0.42 GFLOP/s, indicating that the overhead from tiling, permutation, and predictor operations dominates the computation time, moving the operation further from the roofline ceiling.

**Take-away:** The operational intensity analysis shows that SpMM is compute-bound, but the tiled pipeline's overhead prevents it from reaching the roofline ceiling, highlighting the need for overhead reduction in the tiling strategy.

## Data Reduction Impact

The data reduction achieved through PIM filtering (49.09% reduction in non-zeros) directly impacts the computational characteristics of the pipeline. The filtered dataset (d2) has 3,370 genes compared to the original 33,696 genes, resulting in a 90% reduction in the number of rows. This dramatic reduction changes the tile distribution pattern, with the filtered dataset generating 7,685 tiles (all classified as dense tiles with 43.15% matrix density) compared to the original dataset's tile structure.

The increase in overall matrix density from 8.4% (original) to 43.15% (filtered) after PIM processing suggests that the filtering process retains the more densely populated gene expression patterns while removing sparse noise, which should theoretically benefit GPU acceleration. However, the performance results indicate that the overhead of the tiling and permutation operations outweighs these benefits for this particular dataset size and structure.

**Take-away:** PIM filtering increases matrix density from 8.4% to 43.15%, which should favor GPU processing, but the tiling overhead negates these benefits, indicating a need for adaptive tiling strategies based on dataset characteristics.

