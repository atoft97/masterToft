import cv2
import numpy as np
from os import listdir
from tqdm import tqdm
import os

drivesPath = "../data/combinedImages"

drives = listdir(drivesPath)

#for drive in drives:
#drivePath = drivesPath +"/" + drive
#savePath = "../data/combinedImagesTaller/" + drive
savePath = "/home/potetsos/skule/2022/tmpFiler/segCVATtest2/SegmentationObject"
#loadPath = drivePath
loadPath = "/home/potetsos/skule/2022/tmpFiler/segCVATtest/SegmentationObject"
os.makedirs(savePath, exist_ok=True)
#print("bag: ", drive)
iamgePaths = listdir(loadPath)
iamgePaths.sort()
for imageName in tqdm(iamgePaths):
	originalImage = cv2.imread(loadPath+ "/"+imageName, 1)
	width = originalImage.shape[1] # keep original width
	height = 64*4
	dim = (width, height)


	newImge = cv2.resize(originalImage, dim, interpolation = cv2.INTER_AREA)
	cv2.imwrite(savePath + "/" + imageName, newImge)