import os
import csv
from tqdm import tqdm
from csv import DictWriter
#inputPath = "/home/potetsos/skule/2022/masterCode/masterToft/data/semanticRGB/images"
#iamges = os.listdir(inputPath)
#images.sort()

startPath = "/home/potetsos/lagrinatorn/master/rellisOutput/rgb"
#startPath = "../data/rgbFileName"
trainTestVals =os.listdir(startPath)
print(trainTestVals)

lidarPath = "/home/potetsos/lagrinatorn/master/rellisOutput/lidarDataset"

outputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/RGBtimeLiDARtimeRellis"

outputLidarRGBfilenamePath = "/home/potetsos/lagrinatorn/master/rellisOutput/LidarRGBtimeFilename.csv"


def write_to_csv(new_data, fileName):
    #print(fileName)
    with open(fileName, 'a') as f_object:
        writer_object = DictWriter(f_object, fieldnames=['rgbName', 'lidarName', 'rgbTime', 'lidarTime', 'diffTime'])
        writer_object.writerow(new_data)
        f_object.close()

if (os.path.exists(outputLidarRGBfilenamePath)):
    os.remove(outputLidarRGBfilenamePath)
write_to_csv({'rgbName': 'rgbName', 'lidarName': 'lidarName', 'rgbTime': 'rgbTime', 'lidarTime': 'lidarTime', 'diffTime': 'diffTime'}, outputLidarRGBfilenamePath)

#csvFiles = ["stand_still_short.csv"]

for trainTestVal in trainTestVals:
    fullPath = startPath + "/" + trainTestVal + "/" + "image"
    lidarFullPath = lidarPath + "/" + trainTestVal + "/" + "image"

    print(trainTestVal)

    csvPath = outputPath + "/"+ trainTestVal + ".csv"

    if (os.path.exists(csvPath)):
        os.remove(csvPath)
    write_to_csv({'rgbName': 'rgbName', 'lidarName': 'lidarName', 'rgbTime': 'rgbTime', 'lidarTime': 'lidarTime', 'diffTime':'diffTime'}, csvPath)


    rgbTimes = os.listdir(fullPath)
    rgbTimes.sort()
    lidarTimes = os.listdir(lidarFullPath)
    lidarTimes.sort()

    for rgbTime in tqdm(rgbTimes):
        timeStamp = float(rgbTime[:-4])
        rgbFilename = rgbTime
        #print(timeStamp)
        closest = float("Inf")
        closestLidarFilename = "none"
        for lidarTime in lidarTimes:
            lidar_timeStamp = float(lidarTime[:-4])
            lidarFilename = lidarTime
            #print(abs(int(timeStamp) - int(lidar_timeStamp)) < closest)
            if (abs(float(timeStamp) - float(lidar_timeStamp)) < closest):
                closest = abs(float(timeStamp) - float(lidar_timeStamp))
                closestLidarFilename = lidarFilename
                closesTime = lidar_timeStamp
        #print(closesTime)
        write_to_csv({'rgbName': rgbFilename, 'lidarName': closestLidarFilename, 'rgbTime': timeStamp, 'lidarTime': closesTime, 'diffTime': abs(float(timeStamp)-float(closesTime))}, csvPath)     
        write_to_csv({'rgbName': rgbFilename, 'lidarName': closestLidarFilename, 'rgbTime': timeStamp, 'lidarTime': closesTime, 'diffTime': abs(float(timeStamp)-float(closesTime))}, outputLidarRGBfilenamePath)
            #print(lidar_timeStamp)

