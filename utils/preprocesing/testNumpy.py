import numpy as np
from scipy import sparse



data = np.load("/home/potetsos/lagrinatorn/master/rellisOutput/projectedDataset/train/lidarNumpy/5.npy")

print(data.shape)
#S = sparse.COO(A)
x = sparse.COO(coords, data, shape=((n,) * ndims))
sparceData = sparse.csc_matrix(data)
sparse.save_npz("yourmatrix.npz", sparceData)
your_matrix_back = sparse.load_npz("yourmatrix.npz")

print(your_matrix_back.shape)