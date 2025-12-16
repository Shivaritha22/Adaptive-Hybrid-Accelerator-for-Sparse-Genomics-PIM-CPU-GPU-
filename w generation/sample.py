import numpy as np
import h5py


n_genes = 4
n_cells = 4

# A =
# [1 0 2 0]
# [3 0 4 0]
# [0 0 5 6]
# [0 0 0 7]

indptr  = np.array([0, 2, 2, 5, 7], dtype=np.int64)          # length n_cells + 1
indices = np.array([0, 1, 0, 1, 2, 2, 3], dtype=np.int64)    # row indices (genes)
data    = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)  # values
shape   = np.array([n_genes, n_cells], dtype=np.int64)       # [genes, cells]

# ---------- metadata ----------
barcode_dtype = h5py.string_dtype(encoding="utf-8")
feature_dtype = h5py.string_dtype(encoding="utf-8")

barcodes = np.array(
    ["cell1-1", "cell2-1", "cell3-1", "cell4-1"],
    dtype=barcode_dtype,
)

gene_ids = np.array(
    ["GENE0001", "GENE0002", "GENE0003", "GENE0004"],
    dtype=feature_dtype,
)

gene_names = np.array(
    ["GeneA", "GeneB", "GeneC", "GeneD"],
    dtype=feature_dtype,
)

feature_type = np.array(
    ["Gene Expression"] * n_genes,
    dtype=feature_dtype,
)

genome = np.array(
    ["GRCm39"] * n_genes,
    dtype=feature_dtype,
)

all_tag_keys = np.array(
    ["genome"],
    dtype=feature_dtype,
)

# ---------- write HDF5 in 10x-like layout ----------
with h5py.File("d0.h5", "w") as f:
    g = f.create_group("matrix")

    # main sparse matrix arrays
    g.create_dataset("data",    data=data)
    g.create_dataset("indices", data=indices)
    g.create_dataset("indptr",  data=indptr)
    g.create_dataset("shape",   data=shape)

    # barcodes (cells)
    g.create_dataset("barcodes", data=barcodes)

    # features subgroup (genes)
    feat = g.create_group("features")
    feat.create_dataset("_all_tag_keys", data=all_tag_keys)
    feat.create_dataset("feature_type",  data=feature_type)
    feat.create_dataset("genome",        data=genome)
    feat.create_dataset("id",            data=gene_ids)
    feat.create_dataset("name",          data=gene_names)

print("Sample file created: d0.h5")
