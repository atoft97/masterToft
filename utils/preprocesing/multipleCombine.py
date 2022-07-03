from projectRGBtoLidar import lidarRGBcombine
import os
import csv
from tqdm import tqdm
import cv2
import numpy as np

outputPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetWithRGBCounter"
rgbStartPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetWithRGBCounter"
#rgbLabelStartPath = "/home/potetsos/lagrinatorn/master/rellisOutput/rgb"
lidarStartPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetWithRGBCounter"
lidarPointcloudPath = "/home/potetsos/lagrinatorn/master/Rellis-3D/00002/os1_cloud_node_kitti_bin"

for trainTestVal in ["train", "val", "test"]:
    rgbPath = rgbStartPath + "/" + trainTestVal + "/" + "rgbImage"
    lidarPath = lidarStartPath + "/" + trainTestVal + "/" + "lidarImage"
    lidarNumpyPath = lidarStartPath + "/" + trainTestVal + "/" + "lidarNumpyImage"
    lidarLabelPath = lidarStartPath + "/" + trainTestVal + "/" + "lidarLabelColor"
    lidarLabelIdPath = lidarStartPath + "/" + trainTestVal + "/" + "lidarLabelID"
    lidarCloudPath = lidarStartPath + "/" + trainTestVal + "/" + "lidarPointCloud"

    os.makedirs(f"{outputPath}/{trainTestVal}/rgbProjected/", exist_ok=True)
    os.makedirs(f"{outputPath}/{trainTestVal}/lidarNumpyProjected/", exist_ok=True)
    os.makedirs(f"{outputPath}/{trainTestVal}/lidarLabelColorProjected/", exist_ok=True)
    os.makedirs(f"{outputPath}/{trainTestVal}/lidarLabelIDProjected/", exist_ok=True)
    os.makedirs(f"{outputPath}/{trainTestVal}/projectionVisualised/", exist_ok=True)
    #print(trainTestVal)

    #csvPath = outputPath + "/"+ trainTestVal + ".csv"

    for rgbFilename in tqdm(os.listdir(rgbPath)):

        rgbFullPath = f"{rgbPath}/{rgbFilename}"
        lidarFullPath = f"{lidarPath}/{rgbFilename}"
        lidarNumpyFullPath = f"{lidarNumpyPath}/{rgbFilename[:-4]}.npy"
        lidarLabelFullPath = f"{lidarLabelPath}/{rgbFilename}"
        lidarCloudFullPath = f"{lidarCloudPath}/{rgbFilename[:-4]}.npy"
        lidarLabelIDFullPath = f"{lidarLabelIdPath}/{rgbFilename}"
        #print("path", lidarNumpyFullPath)

        colorImageBackDestaggered, lidarNumpyBackReshaped, lidarLabelBackReshaped, lidarLabelIDBackReshaped, visualised = lidarRGBcombine(lidarCloudFullPath, rgbFullPath, lidarFullPath, lidarNumpyFullPath, lidarLabelFullPath, lidarLabelIDFullPath)
        

        widthCrop = 512
        startX = 1024-245
        endX = 1024+285

        #filter vekk himmel
        #4 gange høyde
        width = 2048
        height = 64*4
        dim = (width, height)

        colorImageBackDestaggered = cv2.resize(colorImageBackDestaggered, dim, interpolation = cv2.INTER_NEAREST)
        lidarNumpyBackReshaped = cv2.resize(lidarNumpyBackReshaped, dim, interpolation = cv2.INTER_NEAREST)
        lidarLabelBackReshaped = cv2.resize(lidarLabelBackReshaped, dim, interpolation = cv2.INTER_NEAREST)
        lidarLabelIDBackReshaped = cv2.resize(lidarLabelIDBackReshaped, dim, interpolation = cv2.INTER_NEAREST)



        colorImageBackDestaggered = colorImageBackDestaggered[:, startX:endX]
        lidarNumpyBackReshaped = lidarNumpyBackReshaped[:, startX:endX]
        lidarLabelBackReshaped = lidarLabelBackReshaped[:, startX:endX]
        lidarLabelIDBackReshaped = lidarLabelIDBackReshaped[:, startX:endX]
        #cv2.imwrite("tmp/test.png", lidarNumpyBackReshaped[:,:,0:3])

        cv2.imwrite(f"{outputPath}/{trainTestVal}/rgbProjected/{rgbFilename}", colorImageBackDestaggered)
        np.save(f"{outputPath}/{trainTestVal}/lidarNumpyProjected/{rgbFilename[:-4]}.npy", lidarNumpyBackReshaped)
        cv2.imwrite(f"{outputPath}/{trainTestVal}/lidarLabelColorProjected/{rgbFilename}", lidarLabelBackReshaped)
        cv2.imwrite(f"{outputPath}/{trainTestVal}/lidarLabelIDProjected/{rgbFilename}", lidarLabelIDBackReshaped)
        cv2.imwrite(f"{outputPath}/{trainTestVal}/projectionVisualised/{rgbFilename}", visualised)
        



    #break
            #except:
           #    print(rgbName)
        #       continue
            #break

                
    #break

#lidarRGBcombine('/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/stand_still_short/stand_still_short00000.npy', '/home/potetsos/lagrinatorn/master/bilder/rgb/stand_still_short/stand_still_short_00000.png', "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedImages/stand_still_short/stand_still_short00001.png", "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/numpyImage/lidar/4.npy", "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/colorSegment/lidar/4.png")from projectLidarRGB import lidarRGBcombine
#import os



#lidarRGBcombine('/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/stand_still_short/stand_still_short00000.npy', '/home/potetsos/lagrinatorn/master/bilder/rgb/stand_still_short/stand_still_short_00000.png')