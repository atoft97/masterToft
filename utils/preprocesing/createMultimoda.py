import os
import cv2
import json
import csv
import shutil

'''
inputpath = "/home/potetsos/lagrinatorn/master/ffiLiDARdataset"
outputPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdataset"
#inputNumpyPath = "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedNumpy/all"

for dataType in ["train", "test", "val"]:
	print(dataType)
	inputpathType = f"{inputpath}/{dataType}/idSegment"
	inputpathTypePaths = os.listdir(inputpathType)
	inputpathTypePaths.sort()
	outputPathNumpy = f"{outputPath}/{dataType}/numpyImage"
	os.makedirs(f"{outputPath}/{dataType}/numpyImage", exist_ok=True)

	for filename in tqdm(inputpathTypePaths):
		#image = cv2.imread(f"{inputpathType}/{filename}")
		fullNumpyPath = f"{inputNumpyPath}/{filename[:-4]}.npy"
		shutil.copyfile(fullNumpyPath, f"{outputPath}/{dataType}/numpyImage/{filename[:-4]}.npy")
'''


rgbImageFolder = "/home/potetsos/lagrinatorn/master/ffiRGBdataset3/all/image"
lidarImageFolder = f"/home/potetsos/skule/2022/masterCode/masterToft/data/combinedImages/all"
lidarNumpyFolder = f"/home/potetsos/skule/2022/masterCode/masterToft/data/combinedNumpy/all"
lidar4channelFolder = "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedNumpy/all"
rgbFolder = "/home/potetsos/lagrinatorn/master/ffiRGBdataset3"
rgbLidarNamePath = "../data/LidarRGBtimeFilename2.csv"
lidarDatasetPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdataset"

outputPath = "/home/potetsos/lagrinatorn/master/multiModal"





with open(rgbLidarNamePath) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')

counter = 0
for dataType in ["train", "test", "val"]:

	os.makedirs(f"{outputPath}/all/rgbImage", exist_ok=True)
	os.makedirs(f"{outputPath}/all/rgbLabelColor", exist_ok=True)
	os.makedirs(f"{outputPath}/all/rgbLabelID", exist_ok=True)

	os.makedirs(f"{outputPath}/all/lidarImage", exist_ok=True)
	os.makedirs(f"{outputPath}/all/lidarLabelColor", exist_ok=True)
	os.makedirs(f"{outputPath}/all/lidarLabelID", exist_ok=True)
	os.makedirs(f"{outputPath}/all/lidarNumpyImage", exist_ok=True)

	os.makedirs(f"{outputPath}/all/lidarPointCloud", exist_ok=True)


	imageNames = os.listdir(f"{lidarDatasetPath}/{dataType}/image")
	imageNames.sort()
	for imageName in imageNames:
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
				if (imageName == lidarFilename):
					rgbImageFullPath = f"{rgbFolder}/all/image/{rgbFilename}"
					rgbLabelIDFullPath = f"{rgbFolder}/all/idSegment/{rgbFilename}"
					rgbLabelColorFullPath = f"{rgbFolder}/all/label/{rgbFilename}"

					lidarImagePath = f"{lidarDatasetPath}/{dataType}/image/{lidarFilename}"
					lidarNumpyImagePath = f"{lidarDatasetPath}/{dataType}/numpyImage/{lidarFilename[:-4]}.npy"
					lidarLabelPath = f"{lidarDatasetPath}/{dataType}/label/{lidarFilename}"
					lidarLabeIDlPath = f"{lidarDatasetPath}/{dataType}/idSegment/{lidarFilename}"

					lidarPoitnCloudPath = f"/home/potetsos/skule/2022/masterCode/masterToft/data/pointCloudAll/{lidarFilename[:-4]}.npy"
					#lidarNumpyFullPath = f"{lidarNumpyFolder}/{lidarFilename[:-4]}.npy"
					if (abs(time1 - time2) > 0.1):
						print("tid:", abs(time1 - time2))
						print("treig", imageName)


					try:
						#print(rgbImageFullPath)
						shutil.copyfile(rgbImageFullPath, f"{outputPath}/all/rgbImage/{counter}.png")
						shutil.copyfile(rgbLabelIDFullPath, f"{outputPath}/all/rgbLabelID/{counter}.png")
						shutil.copyfile(rgbLabelColorFullPath, f"{outputPath}/all/rgbLabelColor/{counter}.png")

						shutil.copyfile(lidarImagePath, f"{outputPath}/all/lidarImage/{counter}.png")
						shutil.copyfile(lidarLabelPath, f"{outputPath}/all/lidarLabelColor/{counter}.png")
						shutil.copyfile(lidarLabeIDlPath, f"{outputPath}/all/lidarLabelID/{counter}.png")
						shutil.copyfile(lidarNumpyImagePath, f"{outputPath}/all/lidarNumpyImage/{counter}.npy")

						shutil.copyfile(lidarPoitnCloudPath, f"{outputPath}/all/lidarPointCloud/{counter}.npy")


						#shutil.copyfile(lidarNumpyFullPath, f"{outputPath}/{dataType}/lidarNumpy/{lidarFilename[:-4]}.npy")
						counter += 1
					except:
						pass
						print("did not find rgbimage:", lidarFilename)
					
					found=True
					break
			if (not found):
				pass
				print("merklig navn", imageName)

	

