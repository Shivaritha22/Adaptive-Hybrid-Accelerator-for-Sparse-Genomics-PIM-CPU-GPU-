# AHAS: Adaptive Hybrid Accelerator for scRNA SpMM

Adaptive hybrid CPU/GPU SpMM pipeline for real 10x scRNA data.  
We tile sparse gene–cell matrices, predict dense vs sparse tiles, route them to cuBLAS (GPU) or CSR+OpenMP (CPU), and explore PIM-style gene filtering to reduce data while preserving key biological signal.

---

## Motivation

Single-cell RNA-seq (scRNA) produces **large, highly sparse** gene–cell matrices.  
A naïve SpMM (`Y = X · W`) is often **memory-bound** and underutilizes the GPU.

This project asks:

- Can we **adaptively route work** between CPU and GPU based on tile density?
- Can we use **PIM-style filtering** (near-storage) to shrink X while keeping biologically important genes?

---

## Objective

1. Build a **tiled, hybrid CPU/GPU SpMM pipeline** for scRNA X and dense W.
2. Use a simple **density-based predictor** to route tiles:
   - Dense tiles → cuBLAS SGEMM on GPU  
   - Sparse tiles → CSR SpMM with OpenMP on CPU
3. Apply an **offline PIM filter** (highly variable genes) to:
   - Reduce nnz(X) and file size  
   - Study impact on performance and effective compute intensity

`K = 32` output features are used for W to:
- Keep a fixed, small output dimension across datasets.
- Make timing and roofline comparisons easier.
- Focus on sparsity and data movement rather than huge dense GEMMs.

---

## Datasets

All X matrices come from public 10x Genomics scRNA datasets:

- **d1:** 10k Mouse Splenocytes (5’ GEM-X)  
  https://www.10xgenomics.com/datasets/10k-Mouse-Splenocytes-5p-gemx

- **d2:** 10k Human DTC Melanoma (NextGEM 5’)  
  https://www.10xgenomics.com/datasets/10k-human-dtc-melanoma-NextGEM-5p

- **d3:** PBMC, granulocytes removed (3k)  
  https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-3-k-1-standard-2-0-0

- **d4:** 30k A549 lung carcinoma cells, CRISPR pool  
  https://www.10xgenomics.com/datasets/30-k-a-549-lung-carcinoma-cells-treatments-transduced-with-a-crispr-pool-multiplexed-6-cm-os-3-1-standard-6-0-0

Convention:

- **X:** cells × genes (on load, converted to internal CSR format as needed)
- **W:** genes × K (here, K = 32)
- **Y:** cells × K

---

## Repository Structure

### `W_generation/`
Generates synthetic dense weight matrices **W** for each X.

- W is sampled from a normal distribution.
- Shape: `genes × 32` to provide a fixed, small output dimension.
- Used across all experiments for consistency.

### `SanityCheck/`
Simple baseline SpMM check.

- Direct CPU SpMM: `Y_ref = X · W` (no tiling, no predictor).
- Serves as:
  - **Correctness reference** for Y.
  - **Baseline timing** for comparisons.

### `scrna/`
Main hybrid pipeline:

- **Tiler:** partitions X into 2D tiles.
- **Predictor:** computes per-tile density and classifies tiles as dense or sparse.
- **SpMM:**
  - Dense tiles → cuBLAS SGEMM on GPU.
  - Sparse tiles → CSR SpMM with OpenMP on CPU.
- Writes Y to HDF5 and compares to reference for unfiltered runs.

### `sanity_cuda/`
CUDA-only sanity checks.

- Small dense GEMM tests to confirm that:
  - cuBLAS calls are wired correctly.
  - Row/column-major layout and transposes are handled properly.

### `roofline/`
Post-processing and analysis.

- Parses logs from the main simulator.
- Computes:
  - FLOPs and bytes moved.
  - Operational intensity (FLOPs/byte).
- Produces data for **roofline-style analysis** across datasets and configurations
  (Baseline vs Hybrid vs PIM+Hybrid).

### `warpspmm/`
Small CUDA experiment (future-work style).

- Standalone warp-level SpMM/GEMM mini-kernel.
- Compares CPU reference vs a warp-shuffle–based GPU kernel.
- Used to explore warp-level programming ideas that could be applied to the main SpMM.

### `PIM/`
Offline PIM-style filtering.

- Uses Scanpy / Python to:
  - Load 10x HDF5 X.
  - Compute per-gene variance across cells.
  - Select **highly variable / “rare” genes**.
- Builds a new, filtered X:
  - Fewer genes and nnz.
  - Preserves high-variance genes as a proxy for important biological signal.
- Filtered X (and corresponding W′) are then fed into the main `scrna/` pipeline.

---

## High-Level Flow

1. Download 10x datasets (X) and generate W (`W_generation/`).
2. Run **SanityCheck** SpMM to get a CPU reference Y.
3. Run **scrna** hybrid pipeline (tiler + predictor + CPU/GPU SpMM).
4. Optionally apply **PIM** filtering, then rerun the hybrid pipeline on filtered X.
5. Use **roofline** tools and logs to analyze performance, intensity, and the effect of PIM.

