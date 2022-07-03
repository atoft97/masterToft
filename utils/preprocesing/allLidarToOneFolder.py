import os
import cv2
import json
import csv
import shutil
from tqdm import tqdm

folderPath = "/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy"
imageFolders = os.listdir(folderPath)
imageFolders.sort()

for imageFolder in imageFolders:
	imageFiles = os.listdir(f"{folderPath}/{imageFolder}")
	imageFiles.sort()
	for imagePath in tqdm(imageFiles):
		fullImagePath = f"{folderPath}/{imageFolder}/{imagePath}"
		shutil.copyfile(fullImagePath, f"/home/potetsos/skule/2022/masterCode/masterToft/data/pointCloudAll/{imagePath}")