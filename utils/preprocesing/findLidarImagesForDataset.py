import os
import cv2
import json
import csv
import shutil

#lidarNumpyFolder = "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedNumpy/all"
#lidarSegmentedFolder = f"{lidarFolder}/SegmentationClass"
lidarImageFolder = f"/home/potetsos/skule/2022/masterCode/masterToft/data/combinedImages/all"
lidarNumpyFolder = f"/home/potetsos/skule/2022/masterCode/masterToft/data/combinedNumpy/all"
#lidarImagePaths = os.listdir(lidarImageFolder)
#lidarImagePaths.sort()

lidar4channelFolder = "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedNumpy/all"
#lidar4channelFilename = os.listdir(lidar4channelFolder)

rgbFolder = "/home/potetsos/lagrinatorn/master/ffiRGBdatasetAndLidar"
#rgbSegmentFolder = f"{rgbFolder}/SegmentationClass"
#rgbImageFolder = f"{rgbFolder}/JPEGImages"

rgbLidarNamePath = "../data/LidarRGBtimeFilename.csv"

outputPath = rgbFolder


#os.makedirs(f"{outputPath}/train/image/rgb", exist_ok=True)
os.makedirs(f"{outputPath}/train/lidarNumpy", exist_ok=True)
#os.makedirs(f"{outputPath}/train/colorSegment/rgb", exist_ok=True)
#os.makedirs(f"{outputPath}/train/colorSegment/lidar", exist_ok=True)
os.makedirs(f"{outputPath}/train/lidarImage", exist_ok=True)

#os.makedirs(f"{outputPath}/test/image/rgb", exist_ok=True)
os.makedirs(f"{outputPath}/test/lidarNumpy", exist_ok=True)
#os.makedirs(f"{outputPath}/test/colorSegment/rgb", exist_ok=True)
#os.makedirs(f"{outputPath}/test/colorSegment/lidar", exist_ok=True)
os.makedirs(f"{outputPath}/test/lidarImage", exist_ok=True)

#os.makedirs(f"{outputPath}/val/image/rgb", exist_ok=True)
os.makedirs(f"{outputPath}/val/lidarNumpy", exist_ok=True)
#os.makedirs(f"{outputPath}/val/colorSegment/rgb", exist_ok=True)
#os.makedirs(f"{outputPath}/val/colorSegment/lidar", exist_ok=True)
os.makedirs(f"{outputPath}/val/lidarImage", exist_ok=True)

#os.makedirs(f"{outputPath}/test", exist_ok=True)
#os.makedirs(f"{outputPath}/val", exist_ok=True)

#trainPath = "/home/potetsos/lagrinatorn/master/cvatLablet/lidarDataset/ImageSets/Segmentation/default.txt"
#f = open(trainPath, "r")

#trainFilenames = []
#trainLidarImagePaths = []

#for line in f:
  #print(line)
#  trainFilenames.append(f"{line.strip()}")
#  trainLidarImagePaths.append(f"{line.strip()}.png")

#print(trainLidarImagePaths)

with open(rgbLidarNamePath) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')

counter = 0
for dataType in ["train", "test", "val"]:

	imageNames = os.listdir(f"{rgbFolder}/{dataType}/image")
	for imageName in imageNames:
		#print(imageName)
		#lidarImageFullPath = f"{lidarImageFolder}/{imageName}"
		#lidarImage = cv2.imread(lidarImageFullPath, -1)

		#lidarSegmentedFullPath = f"{lidarSegmentedFolder}/{lidarImagePath}"
		#lidarImageSegmented = cv2.imread(lidarSegmentedFullPath, -1)

		#lidar4channelFullPath = f"{lidar4channelFolder}/{lidarImagePath[:-4]}.npy"


		with open(rgbLidarNamePath) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')

			for row in csv_reader:
				rgbFilename = row[0]
				lidarFilename = row[1]

				try:
					time1 = float(row[2])
					time2 = float(row[3])
				except:
					pass

				found = False
				if (imageName == rgbFilename):

					#rgbImageFullPath = f"{rgbImageFolder}/{rgbFilename}"
					#rgbImage = cv2.imread(rgbImageFullPath, -1)

					#rgbSegmentImageFullPath = f"{rgbSegmentFolder}/{rgbFilename}"
					#rgbImageSegmented = cv2.imread(rgbSegmentImageFullPath, -1)

					lidarImageFullPath = f"{lidarImageFolder}/{lidarFilename}"
					lidarNumpyFullPath = f"{lidarNumpyFolder}/{lidarFilename[:-4]}.npy"
					#lidarImage = cv2.imread(lidarImageFullPath, -1)

					#print("rgb:", rgbSegmentImageFullPath)
					#print("lidar:", lidarFilename)

					
					#print(lidarImageSegmented.shape)
					#print(rgbImageSegmented.shape)
					#print(rgbImage.shape)
					#print(lidarImage.shape)

					#cv2.imwrite(f"{outputPath}/train/colorSegment/rgb/{counter}.png", rgbImageSegmented)
					#cv2.imwrite(f"{outputPath}/train/colorSegment/lidar/{counter}.png", lidarImageSegmented)

					#cv2.imwrite(f"{outputPath}/train/image/rgb/{counter}.png", rgbImage)
					#cv2.imwrite(f"{outputPath}/train/image/lidar/{counter}.png", lidarImage)
					#print()
					
					if (abs(time1 - time2) > 0.1):
						print("tid:", abs(time1 - time2))
						print("treig", imageName)

					
					try:
						shutil.copyfile(lidarImageFullPath, f"{outputPath}/{dataType}/lidarImage/{lidarFilename}")
						shutil.copyfile(lidarNumpyFullPath, f"{outputPath}/{dataType}/lidarNumpy/{lidarFilename[:-4]}.npy")
						
					except:
						print("did not find lidarimage:", lidarFilename)
					#shutil.copyfile(lidarImageFullPath, f"{outputPath}/{dataType}/numpyImage/{lidarFilename}.npy")
					#cv2.imwrite(f"{outputPath}/train/image/lidar4channel/{counter}.png", lidarImage)
					counter += 1
					found=True
					break
			if (not found):
				print("merklig navn", imageName)

	


	#print(lidarImage.shape)

#rgbPath = "/home/potetsos/lagrinatorn/backupNTNUvm/training/images/stille_frame00027.png"
#outputRgb = "testing/rgbCropped.png"

#lidarPath = "/home/potetsos/Downloads/combinedDatast/images/stand_still_short00024.png"
#outputLidar = "testing/lidarCropped.png"

'''

rgbImage = cv2.imread(rgbPath, -1)
lidarImage = cv2.imread(lidarPath, -1)

#print(rgbImage.shape)


#rgbImage




widthCrop = 512

startX = int(2048/2 - widthCrop/2)
endX = int(2048/2 + widthCrop/2) 

lidarImageCropped = lidarImage[:, startX:endX]
#cv2.imwrite(outputLidar, lidarImageCropped)
#print(lidarImageCropped.shape)


startX = int(2048/2 - widthCrop/2)
endX = int(2048/2 + widthCrop/2) 

rgbImageCropped = rgbImage[1536-1024:1536, :]
#cv2.imwrite(outputRgb, rgbImageCropped)
#print(rgbImageCropped.shape)


height = 512
width = 1024
dim = (width, height)
rgbResized = cv2.resize(rgbImageCropped, dim, interpolation = cv2.INTER_NEAREST)
lidarResized = cv2.resize(lidarImageCropped, dim, interpolation = cv2.INTER_NEAREST)

cv2.imwrite(outputRgb, rgbResized)
cv2.imwrite(outputLidar, lidarResized)

print(rgbResized.shape)
print(lidarResized.shape)
'''