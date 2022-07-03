#import open3d as o3d

import cv2
from ouster import client
from ouster import pcap
import matplotlib.pyplot as plt

import numpy as np

import time
import PIL
import os
from tqdm import tqdm

metadata_path = "lidar_metadata.json"
with open(metadata_path, 'r') as f:
    metadataLidar = client.SensorInfo(f.read())

#path = "Rellis_3D_lidar_example/os1_cloud_node_color_ply/frame000007-1581624653_470.ply"
bagPath = "/home/potetsos/lagrinatorn/master/rellisOutput/lidarImages"
drivePaths = os.listdir(bagPath)
drivePaths.sort()
for drivePath in drivePaths:
    print("Bag:", drivePath)
    drivePathFull = f"{bagPath}/{drivePath}"
    imageTypesPath = os.listdir(drivePathFull)
    for imageTypePath in imageTypesPath:
        print("ImageType:", imageTypePath)
        imageTypePathFull = f"{drivePathFull}/{imageTypePath}"
        imagePaths =  os.listdir(imageTypePathFull)
        imagePaths.sort()
        os.makedirs(f"/home/potetsos/lagrinatorn/master/rellisOutput/destagg/{drivePath}/{imageTypePath}", exist_ok=True)
        for imagePath in tqdm(imagePaths):
            imagePathFull = f"{imageTypePathFull}/{imagePath}"
            inputIamge = cv2.imread(imagePathFull, -1)

            #print(inputIamge.shape)

            rotated = cv2.rotate(inputIamge, cv2.ROTATE_90_CLOCKWISE)
            destaggered = client.destagger(metadataLidar, rotated, inverse=True)
            width = destaggered.shape[1] # keep original width
            height = 64*4
            dim = (width, height)
            rgbTaller = cv2.resize(destaggered, dim, interpolation = cv2.INTER_NEAREST)

            cv2.imwrite(f"/home/potetsos/lagrinatorn/master/rellisOutput/destagg/{drivePath}/{imageTypePath}/{imagePath}", rgbTaller)

            #break
        #break
    #break
'''

folderPath = "/home/potetsos/lagrinatorn/master/rellisOutput/lidarImages/2022-05-09-12-59-41/new_signal_image"
files = os.listdir(folderPath)
files.sort()

for filename in files:
    print(filename)
    filepath = f"{folderPath}/{filename}"
    #pcd = o3d.io.read_point_cloud(filepath)
    #o3d.visualization.draw_geometries([pcd])
    #colors = np.asarray(pcd.colors)
    #colorsReshaped =  colors.reshape(2048, 64, 3) * 255
    inputIamge = cv2.imread(filepath)
    rotated = cv2.rotate(inputIamge, cv2.ROTATE_90_CLOCKWISE)
    destaggered = client.destagger(metadataLidar, rotated, inverse=True)

    width = destaggered.shape[1] # keep original width
    height = 64*4
    dim = (width, height)

    rgbTaller = cv2.resize(destaggered, dim, interpolation = cv2.INTER_NEAREST)
    #rgbTaller = cv2.cvtColor(rgbSmaller.astype('float32'), cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"/home/potetsos/lagrinatorn/master/rellisOutput/destagg/{filename[:-4]}.png", rgbTaller)


'''
