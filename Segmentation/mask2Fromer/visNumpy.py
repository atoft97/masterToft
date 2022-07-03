import numpy as np
import cv2
inputPath = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdataset/train/numpyImage/plains_drive00359.npy"

lidarImage = np.load(inputPath)
cv2.imwrite("heileBilde.png",lidarImage[:,:,0:3])