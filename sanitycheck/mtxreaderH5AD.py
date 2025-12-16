import sys, scanpy as sc, scipy.sparse as sp
import pandas as pd
from scipy.io import mmwrite
fn = sys.argv[1]
ad = sc.read(fn) 
X = ad.X.tocsr() if not sp.isspmatrix_csr(ad.X) else ad.X
mmwrite("matrix.mtx", X) 
pd.Series(ad.var_names).to_csv("features.tsv", index=False, header=False)
pd.Series(ad.obs_names).to_csv("barcodes.tsv", index=False, header=False)
print("Files ready: matrix.mtx, features.tsv, barcodes.tsv")
