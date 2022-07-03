import cv2
import numpy as np
from os import listdir
from tqdm import tqdm
import os

drivesPath = "/home/potetsos/lagrinatorn/master/rellisOutput/destagg"
drives = listdir(drivesPath)

for drive in drives:
	drivePath = drivesPath +"/" + drive
	savePath = drivePath + "/new_range_bright"
	loadPath = drivePath + "/new_range_image"
	os.makedirs(savePath, exist_ok=True)
	print("bag: ", drive)
	iamgePaths = listdir(drivesPath +"/" + drive + "/new_range_image")
	iamgePaths.sort()
	for imageName in tqdm(iamgePaths):
		originalImage = cv2.imread(loadPath+ "/"+imageName, -1)
		newImge = cv2.multiply(originalImage, 10) #multply by 15
		#newImge[newImge <= 0] = 255 #caps at 255
		#newImge = cv2.bitwise_not(newImge)
		#print(originalImage)
		#print(originalImage.shape)

		cv2.imwrite(savePath + "/" + imageName, newImge)

		#cv2.imwrite("test1New.png", newImge)
		#print(newImge)
		#bilde = cv2.imread("test1New.png", 0)
		#print(bilde)

		#cv2.imwrite("test2New.png", bilde)
		#print(bilde)
		
	