import sys, os, scanpy as sc, scipy.sparse as sp
import pandas as pd
from scipy.io import mmwrite

fn = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(fn))[0]

ad = sc.read_10x_h5(fn)
ad.var_names_make_unique()

X = ad.X if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
X = X.tocsr() if not sp.isspmatrix_csr(X) else X
if X.shape == (ad.n_vars, ad.n_obs):  # ensure cells × genes
    X = X.T.tocsr()

mmwrite(f"{base}_matrix.mtx", X)
pd.Series(ad.var_names).to_csv(f"{base}_features.tsv", index=False, header=False)
pd.Series(ad.obs_names).to_csv(f"{base}_barcodes.tsv", index=False, header=False)
print(f"Files ready: {base}_matrix.mtx ({X.shape[0]}x{X.shape[1]}), {base}_features.tsv, {base}_barcodes.tsv")
