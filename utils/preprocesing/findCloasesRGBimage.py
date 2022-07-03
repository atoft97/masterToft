import os
import csv
from tqdm import tqdm
from csv import DictWriter
#inputPath = "/home/potetsos/skule/2022/masterCode/masterToft/data/semanticRGB/images"
#iamges = os.listdir(inputPath)
#images.sort()

startPath = "../data/rgbFileName2"
#startPath = "../data/rgbFileName"
csvFiles =os.listdir(startPath)
print(csvFiles)

lidarPath = "../data/lidarFilename"

outputPath = "../data/RGBtimeLiDARtime2"

outputLidarRGBfilenamePath = "../data/LidarRGBtimeFilename2.csv"


def write_to_csv(new_data, fileName):
    with open(fileName, 'a') as f_object:
        writer_object = DictWriter(f_object, fieldnames=['rgbName', 'lidarName', 'rgbTime', 'lidarTime', 'diffTime'])
        writer_object.writerow(new_data)
        f_object.close()

if (os.path.exists(outputLidarRGBfilenamePath)):
    os.remove(outputLidarRGBfilenamePath)
write_to_csv({'rgbName': 'rgbName', 'lidarName': 'lidarName', 'rgbTime': 'rgbTime', 'lidarTime': 'lidarTime', 'diffTime': 'diffTime'}, outputLidarRGBfilenamePath)

#csvFiles = ["stand_still_short.csv"]

for csvFile in csvFiles:
    fullPath = startPath + "/" + csvFile
    lidarFullPath = lidarPath + "/" + csvFile

    print(csvFile)

    csvPath = outputPath + "/"+ csvFile

    if (os.path.exists(csvPath)):
        os.remove(csvPath)
    write_to_csv({'rgbName': 'rgbName', 'lidarName': 'lidarName', 'rgbTime': 'rgbTime', 'lidarTime': 'lidarTime', 'diffTime':'diffTime'}, csvPath)



    with open(fullPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rgbCounter = 0
        for row in tqdm(csv_reader):
            if (rgbCounter >1):
                timeStamp = row[0]
                rgbFilename = row[1]
                #print(timeStamp)
                closest = float("Inf")
                closestLidarFilename = "none"
                with open(lidarFullPath) as lidar_csv:
                    lidar_csv_reader = csv.reader(lidar_csv, delimiter=',')
                    lidarCounter = 0
                    for lidar_row in lidar_csv_reader:
                        if (lidarCounter > 1):
                            lidar_timeStamp = lidar_row[0]
                            lidarFilename = lidar_row[1]
                            #print(abs(int(timeStamp) - int(lidar_timeStamp)) < closest)
                            if (abs(float(timeStamp) - float(lidar_timeStamp)) < closest):
                                closest = abs(float(timeStamp) - float(lidar_timeStamp))
                                closestLidarFilename = lidarFilename
                                closesTime = lidar_timeStamp
                        lidarCounter += 1
                #print(closesTime)
                write_to_csv({'rgbName': rgbFilename, 'lidarName': closestLidarFilename, 'rgbTime': timeStamp, 'lidarTime': closesTime, 'diffTime': abs(float(timeStamp)-float(closesTime))}, csvPath)     
                write_to_csv({'rgbName': rgbFilename, 'lidarName': closestLidarFilename, 'rgbTime': timeStamp, 'lidarTime': closesTime, 'diffTime': abs(float(timeStamp)-float(closesTime))}, outputLidarRGBfilenamePath)
            rgbCounter += 1
                    #print(lidar_timeStamp)

