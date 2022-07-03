import os
import cv2
import numpy as np
datasetPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetWithRGBCounter"

outputPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetWithRGBCounter"

filenames = os.listdir(f"{datasetPath}/")

height = 512
width = 1024
dim = (width, height)

for dataType in ["train", "test", "val"]:
	for folder in os.listdir(f"{datasetPath}/{dataType}"):
		for sensor in os.listdir(f"{datasetPath}/{dataType}/{folder}"):
			os.makedirs(f"{outputPath}/{dataType}/{folder}/{sensor}", exist_ok=True)
			if (sensor == "rgb"):
				for imageName in os.listdir(f"{datasetPath}/{dataType}/{folder}/{sensor}"):
					rgbPath = f"{datasetPath}/{dataType}/{folder}/{sensor}/{imageName}"
					rgbImage = cv2.imread(rgbPath, -1)
					#print(rgbImage.shape)
					rgbImageCropped = rgbImage[1536-1024:1536, :]

					rgbResized = cv2.resize(rgbImageCropped, dim, interpolation = cv2.INTER_NEAREST)
					cv2.imwrite(f"{outputPath}/{dataType}/{folder}/{sensor}/{imageName}", rgbResized)
			if (sensor == "lidar"):
				for imageName in os.listdir(f"{datasetPath}/{dataType}/{folder}/{sensor}"):
					#print(imageName[-3:])
					if (imageName[-3:] == "npy"):
						lidarPath = f"{datasetPath}/{dataType}/{folder}/{sensor}/{imageName}"
						lidarNumpy = np.load(lidarPath)

						widthCrop = 512
						startX = int(2048/2 - widthCrop/2)
						endX = int(2048/2 + widthCrop/2) 
						lidarNumpyCropped = lidarNumpy[:, startX:endX]

						#print(lidarNumpyCropped.shape)
						lidarNumpyResized = cv2.resize(lidarNumpyCropped, dim, interpolation = cv2.INTER_NEAREST)
						#print(lidarNumpyResized.shape)
						#lidarImage = cv2.imread(lidarPath, -1)
						#print(lidarImage.shape)
						#cv2.imwrite()
						np.save(f"{outputPath}/{dataType}/{folder}/{sensor}/{imageName}", lidarNumpyResized)

					else:
						lidarPath = f"{datasetPath}/{dataType}/{folder}/{sensor}/{imageName}"
						lidarImage = cv2.imread(lidarPath, -1)
						#print(rgbImage.shape)
						widthCrop = 512
						startX = int(2048/2 - widthCrop/2)
						endX = int(2048/2 + widthCrop/2) 
						lidarImageCropped = lidarImage[:, startX:endX]

						lidarResized = cv2.resize(lidarImageCropped, dim, interpolation = cv2.INTER_NEAREST)
						cv2.imwrite(f"{outputPath}/{dataType}/{folder}/{sensor}/{imageName}", lidarResized)


		#outputRgb = "testing/rgbCropped.png"

		#lidarPath = "/home/potetsos/Downloads/combinedDatast/images/stand_still_short00024.png"
		#outputLidar = "testing/lidarCropped.png"

def reziseLidar()