import numpy as np

numpyPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdataset16/train/numpyImage/plains_drive00541.npy"

numpyBilde = np.load(numpyPath)

print(numpyBilde[:,:,3])