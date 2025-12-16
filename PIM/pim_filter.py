import os
import sys
import numpy as np
import scanpy as sc
import h5py
from pathlib import Path
from scipy import sparse

def load_h5_file(filepath):
    """Load 10x h5 file using Scanpy"""
    try:
        adata = sc.read_10x_h5(filepath)
        # Transpose to get genes x cells (if needed, Scanpy usually stores as cells x genes)
        # Actually, Scanpy stores as cells x genes, so adata.X is cells x genes
        # For variance computation, we need genes x cells, so we'll work with adata.X.T
        return adata
    except Exception as e:
        print(f"Error loading h5 file: {e}")
        return None

def compute_gene_variance(adata):
    """Compute variance per gene over all cells"""
    # adata.X is cells x genes (sparse or dense)
    # Compute variance along axis 0 (across cells) for each gene
    if hasattr(adata.X, 'toarray'):
        # Sparse matrix
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # X is cells x genes, so variance along axis 0 gives variance per gene
    # Ensure we get a 1D array (flatten if needed)
    gene_variances = np.var(X, axis=0)
    if gene_variances.ndim > 1:
        gene_variances = gene_variances.flatten()
    
    return gene_variances

def compute_gene_mean(adata):
    """Compute mean expression per gene over all cells"""
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # Ensure we get a 1D array (flatten if needed)
    gene_means = np.mean(X, axis=0)
    if gene_means.ndim > 1:
        gene_means = gene_means.flatten()
    
    return gene_means

def identify_rare_genes(gene_variances, top_percent=0.1):
    """Identify top N% most variable genes (rare / HVG genes)"""
    n_genes = len(gene_variances)
    # Use round to handle fractional percentages properly
    n_top = max(1, int(round(n_genes * top_percent)))

    # Get indices sorted by variance (descending)
    sorted_indices = np.argsort(gene_variances)[::-1]
    rare_gene_indices = sorted_indices[:n_top]

    return rare_gene_indices


def identify_noise_genes(gene_variances, bottom_percent=0.1):
    """Identify bottom N% least variable genes (noise genes) - excluding zero variance genes"""
    n_genes = len(gene_variances)
    
    # Filter out zero-variance genes (they're not informative for noise detection)
    non_zero_var_mask = gene_variances > 0
    non_zero_indices = np.where(non_zero_var_mask)[0]
    
    if len(non_zero_indices) == 0:
        # All genes have zero variance - return empty
        return np.array([], dtype=int)
    
    # Work with non-zero variance genes only
    non_zero_variances = gene_variances[non_zero_indices]
    n_valid = len(non_zero_variances)
    
    # Calculate bottom percentage of valid (non-zero variance) genes
    n_bottom = max(1, int(round(n_valid * bottom_percent)))
    
    # Get indices sorted by variance (ascending) among non-zero variance genes
    sorted_local_indices = np.argsort(non_zero_variances)
    noise_local_indices = sorted_local_indices[:n_bottom]
    
    # Map back to original gene indices
    noise_gene_indices = non_zero_indices[noise_local_indices]

    return noise_gene_indices


def identify_housekeeping_genes(gene_means, gene_variances, top_mean_percent=0.2, low_var_percent=0.3):
    """Identify housekeeping genes: high mean expression but low variance"""
    n_genes = len(gene_means)
    
    # Filter out genes with zero variance or zero mean (they're not informative)
    valid_mask = (gene_variances > 0) & (gene_means > 0)
    if np.sum(valid_mask) == 0:
        # No valid genes - return empty
        return np.array([], dtype=int)
    
    # Work with genes that have non-zero variance and non-zero mean
    valid_indices = np.where(valid_mask)[0]
    valid_means = gene_means[valid_indices]
    valid_variances = gene_variances[valid_indices]
    
    n_valid = len(valid_indices)
    if n_valid == 0:
        return np.array([], dtype=int)
    
    # Housekeeping genes: HIGH mean expression AND LOW variance
    # This is different from noise genes which have low variance but might have low mean
    
    # Get top genes by mean expression (highly expressed)
    n_top_mean = max(1, int(round(n_valid * top_mean_percent)))
    sorted_by_mean = np.argsort(valid_means)[::-1]  # Descending
    top_mean_local_indices = sorted_by_mean[:n_top_mean]
    top_mean_indices = set(valid_indices[top_mean_local_indices])
    
    # Get bottom genes by variance (low variance = stable)
    n_low_var = max(1, int(round(n_valid * low_var_percent)))
    sorted_by_var = np.argsort(valid_variances)  # Ascending
    low_var_local_indices = sorted_by_var[:n_low_var]
    low_var_indices = set(valid_indices[low_var_local_indices])
    
    # Housekeeping = high mean AND low variance (intersection)
    housekeeping_indices = np.array(list(top_mean_indices & low_var_indices))
    
    # If intersection is too small, relax criteria slightly
    if len(housekeeping_indices) < 100:
        # Try with slightly relaxed criteria
        n_top_mean_relaxed = max(1, int(round(n_valid * (top_mean_percent + 0.1))))
        n_low_var_relaxed = max(1, int(round(n_valid * (low_var_percent + 0.1))))
        top_mean_indices_relaxed = set(valid_indices[sorted_by_mean[:n_top_mean_relaxed]])
        low_var_indices_relaxed = set(valid_indices[sorted_by_var[:n_low_var_relaxed]])
        housekeeping_indices = np.array(list(top_mean_indices_relaxed & low_var_indices_relaxed))
    
    return housekeeping_indices

def extract_annotation_from_filename(filename):
    """Extract annotation identifier from filename (e.g., 'd2' from 'd2.h5')"""
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Extract annotation (assuming format like 'd2', 'w0', etc.)
    return name

def save_pim_filter(output_path, adata, rare_gene_indices):
    """Save PIM filtered matrix to 10x h5 file format (filtered version of original data)"""
    
    # Sort indices to maintain order
    rare_gene_indices_sorted = np.sort(rare_gene_indices)
    
    # Filter the AnnData object to keep only rare genes
    adata_filtered = adata[:, rare_gene_indices_sorted].copy()
    
    # --- MATRIX: build genes x cells in CSC (10x style) ---
    # adata_filtered.X is cells x genes
    X = adata_filtered.X

    # We want a CSC matrix of shape (genes, cells) so that:
    #   shape[0] = n_genes_kept
    #   shape[1] = n_cells
    #   indptr length = n_cells + 1
    if sparse.isspmatrix(X):
        # X is (cells x genes); transpose to (genes x cells).
        # Transpose of CSR is CSC, so this gives us CSC with correct indptr.
        X_csc = X.T.tocsc()
    else:
        # Dense -> convert after transpose
        X_csc = sparse.csc_matrix(X.T)

    # X_csc now has:
    #   X_csc.shape = (n_genes_kept, n_cells)
    #   X_csc.data, X_csc.indices, X_csc.indptr in CSC layout

    # --- BARCODES (cells) ---
    barcodes = adata_filtered.obs.index.values
    if barcodes.dtype != 'object':
        barcodes = barcodes.astype(str)
    barcodes_bytes = np.array([bc.encode('utf-8') for bc in barcodes])

    # --- FEATURES (genes) ---
    feature_names = adata_filtered.var.index.values
    if feature_names.dtype != 'object':
        feature_names = feature_names.astype(str)
    feature_names_bytes = np.array([name.encode('utf-8') for name in feature_names])

    # Feature IDs (use gene_ids if available, else names)
    if 'gene_ids' in adata_filtered.var.columns:
        feature_ids = adata_filtered.var['gene_ids'].values
        if feature_ids.dtype != 'object':
            feature_ids = feature_ids.astype(str)
        feature_ids_bytes = np.array([fid.encode('utf-8') for fid in feature_ids])
    else:
        feature_ids_bytes = feature_names_bytes.copy()

    # Feature types
    if 'feature_type' in adata_filtered.var.columns:
        feature_types = adata_filtered.var['feature_type'].values
        if feature_types.dtype != 'object':
            feature_types = feature_types.astype(str)
        feature_types_bytes = np.array([ftype.encode('utf-8') for ftype in feature_types])
    else:
        feature_types_bytes = np.array([b'Gene Expression'] * len(feature_names))

    # Genome annotation
    if 'genome' in adata_filtered.var.columns:
        genomes = adata_filtered.var['genome'].values
        if genomes.dtype != 'object':
            genomes = genomes.astype(str)
        genome_bytes = np.array([g.encode('utf-8') for g in genomes])
    else:
        if hasattr(adata, 'var') and 'genome' in adata.var.columns:
            genomes = adata.var.loc[adata_filtered.var.index, 'genome'].values
            if genomes.dtype != 'object':
                genomes = genomes.astype(str)
            genome_bytes = np.array([g.encode('utf-8') for g in genomes])
        else:
            genome_bytes = np.array([b''] * len(feature_names))

    # --- WRITE H5 IN 10x-LIKE FORMAT ---
    with h5py.File(output_path, 'w') as f:
        matrix_group = f.create_group('matrix')

        # barcodes: length = n_cells
        try:
            matrix_group.create_dataset(
                'barcodes',
                data=barcodes_bytes,
                dtype=h5py.string_dtype(encoding='utf-8')
            )
        except (AttributeError, TypeError):
            dt = h5py.special_dtype(vlen=str)
            barcodes_str = [b.decode('utf-8') for b in barcodes_bytes]
            matrix_group.create_dataset('barcodes', data=barcodes_str, dtype=dt)

        # sparse matrix data in CSC (genes x cells)
        matrix_group.create_dataset('data',    data=X_csc.data.astype('float32'),  dtype='float32')
        matrix_group.create_dataset('indices', data=X_csc.indices.astype('int32'), dtype='int32')
        matrix_group.create_dataset('indptr',  data=X_csc.indptr.astype('int32'),  dtype='int32')
        matrix_group.create_dataset(
            'shape',
            data=np.array([X_csc.shape[0], X_csc.shape[1]], dtype='int64')
        )

        # features group
        features_group = matrix_group.create_group('features')

        def save_string_dataset(group, name, bytes_data):
            try:
                group.create_dataset(name, data=bytes_data,
                                     dtype=h5py.string_dtype(encoding='utf-8'))
            except (AttributeError, TypeError):
                dt = h5py.special_dtype(vlen=str)
                str_data = [b.decode('utf-8') for b in bytes_data]
                group.create_dataset(name, data=str_data, dtype=dt)

        save_string_dataset(features_group, 'id',           feature_ids_bytes)
        save_string_dataset(features_group, 'name',         feature_names_bytes)
        save_string_dataset(features_group, 'feature_type', feature_types_bytes)
        save_string_dataset(features_group, 'genome',       genome_bytes)

        # extra metadata (safe – readers ignore unknown attrs)
        matrix_group.attrs['rare_gene_indices']  = rare_gene_indices_sorted
        matrix_group.attrs['n_rare_genes']       = len(rare_gene_indices_sorted)
        matrix_group.attrs['n_original_genes']   = adata.n_vars
        matrix_group.attrs['n_cells']            = adata.n_obs


def save_meta_file(meta_path, n_noise, n_rare, n_housekeep, gene_variances):
    """Save metadata to text file"""
    with open(meta_path, 'w') as f:
        f.write(f"noise gene: {n_noise}\n")
        f.write(f"rare gene: {n_rare}\n")
        f.write(f"common housekeep: {n_housekeep}\n")
        f.write(f"\n")
        f.write(f"variance min: {np.min(gene_variances):.6f}\n")
        f.write(f"variance max: {np.max(gene_variances):.6f}\n")
        f.write(f"variance mean: {np.mean(gene_variances):.6f}\n")
        f.write(f"genes with zero variance: {np.sum(gene_variances == 0)}\n")
        f.write(f"unique variance values: {len(np.unique(gene_variances))}\n")

def main():
    """Main function"""
    # Get filename from terminal
    if len(sys.argv) < 2:
        print("Usage: python pim_filter.py <filename>")
        print("Example: python pim_filter.py d2.h5")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Define paths
    base_path = Path(__file__).parent
    original_path = base_path / "original/X"
    input_file = original_path / filename
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Loading file: {input_file}")
    
    # Load h5 file
    adata = load_h5_file(str(input_file))
    if adata is None:
        print("Error: Failed to load h5 file")
        sys.exit(1)
    
    print(f"Loaded data: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Compute variance per gene
    print("Computing gene variances...")
    gene_variances = compute_gene_variance(adata)
    gene_means = compute_gene_mean(adata)
    
    # Debug: Check variance distribution
    print(f"\nVariance statistics:")
    print(f"  Min variance: {np.min(gene_variances):.6f}")
    print(f"  Max variance: {np.max(gene_variances):.6f}")
    print(f"  Mean variance: {np.mean(gene_variances):.6f}")
    print(f"  Genes with zero variance: {np.sum(gene_variances == 0)}")
    print(f"  Unique variance values: {len(np.unique(gene_variances))}")
    
    # Identify gene categories
    print("\nIdentifying gene categories...")
    rare_gene_indices = identify_rare_genes(gene_variances, top_percent=0.1)
    noise_gene_indices = identify_noise_genes(gene_variances, bottom_percent=0.1)
    housekeeping_indices = identify_housekeeping_genes(gene_means, gene_variances)
    
    # Ensure housekeeping genes are distinct from noise genes
    # Housekeeping should have high mean, noise might have low mean
    noise_set = set(noise_gene_indices)
    housekeeping_indices = np.array([idx for idx in housekeeping_indices if idx not in noise_set])
    
    # Check for overlaps
    overlap_rare_noise = np.intersect1d(rare_gene_indices, noise_gene_indices)
    overlap_rare_housekeep = np.intersect1d(rare_gene_indices, housekeeping_indices)
    overlap_noise_housekeep = np.intersect1d(noise_gene_indices, housekeeping_indices)
    
    if len(overlap_rare_noise) > 0:
        print(f"WARNING: {len(overlap_rare_noise)} genes appear in both rare and noise sets!")
    if len(overlap_rare_housekeep) > 0:
        print(f"INFO: {len(overlap_rare_housekeep)} genes appear in both rare and housekeeping sets (possible)")
    if len(overlap_noise_housekeep) > 0:
        print(f"WARNING: {len(overlap_noise_housekeep)} genes appear in both noise and housekeeping sets (removed from housekeeping)")
    
    print(f"\nGene category counts:")
    print(f"  Rare genes (top 10% variance): {len(rare_gene_indices)}")
    print(f"  Noise genes (bottom 10% variance): {len(noise_gene_indices)}")
    print(f"  Housekeeping genes (high mean, low variance): {len(housekeeping_indices)}")
    
    # Extract annotation from filename
    annotation = extract_annotation_from_filename(filename)
    
    # Save outputs
    output_h5 = base_path / f"{annotation}_pim.h5"
    meta_file = base_path / f"{annotation}_meta.txt"
    
    print(f"\nSaving PIM filtered matrix to: {output_h5}")
    print(f"  Original: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Filtered: {adata.n_obs} cells x {len(rare_gene_indices)} genes")
    save_pim_filter(str(output_h5), adata, rare_gene_indices)
    
    print(f"Saving metadata to: {meta_file}")
    save_meta_file(str(meta_file), len(noise_gene_indices), len(rare_gene_indices), len(housekeeping_indices), gene_variances)
    
    print("\nDone!")
    print(f"Output files:")
    print(f"  - {output_h5}")
    print(f"  - {meta_file}")

if __name__ == "__main__":
    main()

