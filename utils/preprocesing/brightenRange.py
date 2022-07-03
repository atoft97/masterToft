import cv2
import numpy as np
from os import listdir
from tqdm import tqdm
import os

drivesPath = "../data/lidarImages16"
drives = listdir(drivesPath)

'''
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
		newImge = cv2.multiply(originalImage, 15) #multply by 15
		newImge[newImge <= 0] = 255 #caps at 255
		newImge = cv2.bitwise_not(newImge)
		cv2.imwrite(savePath + "/" + imageName, newImge)
'''
#drives = ["plains_drive"]
for drive in drives:
	drivePath = drivesPath +"/" + drive
	savePath = drivePath + "/rangeBrightWhole"
	loadPath = drivePath + "/range_image"
	os.makedirs(savePath, exist_ok=True)
	print("bag: ", drive)
	iamgePaths = listdir(drivesPath +"/" + drive + "/range_image")
	iamgePaths.sort()
	for imageName in tqdm(iamgePaths):
		originalImage = cv2.imread(loadPath+ "/"+imageName, -1)
		#print(originalImage)
		#print(originalImage.shape)
		newImge = cv2.multiply(originalImage, 1) #multply by 15
		#newImge[newImge <= 0] = 255*255 #caps at 255
		#newImge = cv2.bitwise_not(newImge)
		cv2.imwrite(savePath + "/" + imageName, newImge)

		#cv2.imwrite("test1.png", newImge)
		#print(newImge)
		#bilde = cv2.imread("test1.png", 0)
		#print(bilde)

		#cv2.imwrite("test2.png", bilde)
		#print(bilde)
		
	
		
		