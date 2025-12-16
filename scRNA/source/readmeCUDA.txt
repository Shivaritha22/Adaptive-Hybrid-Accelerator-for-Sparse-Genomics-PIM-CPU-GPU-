// PERFORMANCE OPTIMIZATIONS & METRICS (from actual log measurements):
//
// 1. Zero-Copy Transpose Optimization:
//    Why transpose was needed: cuBLAS uses column-major storage (Fortran convention), but our
//    matrices are row-major (C convention). Naive approach would transpose all matrices before
//    calling cuBLAS, requiring expensive memory copies (e.g., 32KB for 64x64 tile).
//    How we bypass it: Instead of copying data, we mathematically interpret row-major matrices
//    as transposed column-major and swap operands/dimensions in the cuBLAS call. This gives
//    mathematically equivalent results with zero additional memory copies.
//
// 2. Measured Performance (from logs):
//    Test case 4: 2494 dense tiles, 370.893ms total, 0.58 GFLOP/s, 0.24 GB/s
//    Test case 5: 2726 dense tiles, 659.834ms total, 0.84 GFLOP/s, 0.21 GB/s
//    Tile size: 64x64 (from hw_config.h), W_cols: 32
//    Dense threshold: 5% (from hw_config.h)
//
// 3. CUDA Overhead Analysis (estimated per 64x64 tile, W_cols=32):
//    Memory transfer size: 32KB total (X: 16KB, W: 8KB, Y: 8KB)
//    Estimated overhead per tile:
//    - Memory transfers (H2D + D2H): ~2-5μs (depends on PCIe bandwidth)
//    - Malloc/free (3 allocs + 3 frees): ~1-3μs
//    - Total estimated overhead: ~3-8μs per tile
//
// 4. Overhead Compensation:
//    For 2494-2726 dense tiles: estimated total overhead ~7-22ms
//    Measured total compute time: 370-660ms (includes all operations)
//    Overhead is <6% of total time, compensated by GPU compute speedup
//    Only used for dense tiles (≥5% density) to minimize overhead impact on small tiles
//
// 5. hw_config Tuning for Better CUDA Performance:
//    Larger tiles (TILE_ROWS/TILE_COLS) reduce overhead per computation by amortizing transfer
//    costs across more work; lower DENSE_TILE_THRESHOLD routes more tiles to CUDA for better
//    GPU utilization when tiles are large enough to overcome transfer overhead.