from projectLidarRGBrellis import lidarRGBcombine
import os
import csv
from tqdm import tqdm

outputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/RGBtimeLiDARtimeRellis"
rgbStartPath = "/home/potetsos/lagrinatorn/master/rellisOutput/rgb"
rgbLabelStartPath = "/home/potetsos/lagrinatorn/master/rellisOutput/rgb"
lidarStartPath = "/home/potetsos/lagrinatorn/master/rellisOutput/lidarDataset"
lidarPointcloudPath = "/home/potetsos/lagrinatorn/master/Rellis-3D/00002/os1_cloud_node_kitti_bin"

for trainTestVal in ["val", "test"]:
    rgbPath = rgbStartPath + "/" + trainTestVal + "/" + "image"
    lidarPath = lidarStartPath + "/" + trainTestVal + "/" + "image"
    lidarNumpyPath = lidarStartPath + "/" + trainTestVal + "/" + "numpyImage"
    lidarLabelPath = lidarStartPath + "/" + trainTestVal + "/" + "label"
    lidarCloudPath = lidarStartPath + "/" + trainTestVal + "/" + "numpyCloud"

    #print(trainTestVal)

    csvPath = outputPath + "/"+ trainTestVal + ".csv"

    with open(csvPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rgbCounter = 0
        for row in tqdm(csv_reader):
            if (950>rgbCounter >1):
                rgbName = row[0]
                lidarName = row[1]

                rgbFullPath = f"{rgbPath}/{rgbName}"
                lidarFullPath = f"{lidarPath}/{lidarName}"
                lidarNumpyFullPath = f"{lidarNumpyPath}/{lidarName[:-4]}.npy"
                lidarLabelFullPath = f"{lidarLabelPath}/{lidarName}"
                lidarCloudFullPath = f"{lidarCloudPath}/{lidarName[:-4]}.npy"
                #try:
                #print(rgbFullPath)
                #print(lidarFullPath)

                #print(lidarNumpyFullPath)

                #print(lidarLabelFullPath)

                #print(lidarCloudFullPath)


                lidarRGBcombine(lidarCloudFullPath, rgbFullPath, lidarFullPath, lidarNumpyFullPath, lidarLabelFullPath, trainTestVal, rgbCounter)
                #except:
               # 	print(rgbName)
            #		continue
                #break
            rgbCounter += 1
                
    #break

#lidarRGBcombine('/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/stand_still_short/stand_still_short00000.npy', '/home/potetsos/lagrinatorn/master/bilder/rgb/stand_still_short/stand_still_short_00000.png', "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedImages/stand_still_short/stand_still_short00001.png", "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/numpyImage/lidar/4.npy", "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/colorSegment/lidar/4.png")