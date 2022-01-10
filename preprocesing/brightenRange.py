import cv2
import numpy as np
from os import listdir
from tqdm import tqdm
import os

drivesPath = "../data/lidarImages"
drives = listdir(drivesPath)

for drive in drives:
	drivePath = drivesPath +"/" + drive
	savePath = drivePath + "/rangeBright"
	loadPath = drivePath + "/range_image"
	os.makedirs(savePath, exist_ok=True)
	print("bag: ", drive)
	iamgePaths = listdir(drivesPath +"/" + drive + "/range_image")
	iamgePaths.sort()
	for imageName in tqdm(iamgePaths):
		originalImage = cv2.imread(loadPath+ "/"+imageName, 0)
		
		newImge = cv2.multiply(originalImage, 15)


		newImge[newImge <= 0] = 255

		newImge = cv2.bitwise_not(newImge)

		cv2.imwrite(savePath + "/" + imageName, newImge)