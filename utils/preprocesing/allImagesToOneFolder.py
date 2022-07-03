import os
import cv2
import json
import csv
import shutil
from tqdm import tqdm

folderPath = "/home/potetsos/lagrinatorn/master/bilder/rgb"
imageFolders = os.listdir(folderPath)
imageFolders.sort()

for imageFolder in imageFolders:
	if (imageFolder != "all"):
		imageFiles = os.listdir(f"{folderPath}/{imageFolder}")
		imageFiles.sort()
		for imagePath in tqdm(imageFiles):
			fullImagePath = f"{folderPath}/{imageFolder}/{imagePath}"
			shutil.copyfile(fullImagePath, f"{folderPath}/all/{imagePath}")