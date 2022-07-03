import open3d as o3d

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


folderPath = "/home/potetsos/lagrinatorn/master/Rellis-3D"
drives = os.listdir(folderPath)
drives.sort()

for drive in drives:
    print("drive:", drive)
    filenames = os.listdir(f"{folderPath}/{drive}/os1_cloud_node_color_ply")
    filenames.sort()
    #os.makedirs(f"/home/potetsos/lagrinatorn/master/rellisOutput/numpyPointcloud/{drive}", exist_ok=True)
    os.makedirs(f"/home/potetsos/lagrinatorn/master/rellisOutput/numpyColorsCloud/{drive}", exist_ok=True)
    for filename in tqdm(filenames):
        filepath = f"{folderPath}/{drive}/os1_cloud_node_color_ply/{filename}"
        pcd = o3d.io.read_point_cloud(filepath)
        #o3d.visualization.draw_geometries([pcd])
        colors = np.asarray(pcd.colors)
        points = np.asarray(pcd.points)

        colorsReshaped =  colors.reshape(2048, 64, 3) * 255
        rotated = cv2.rotate(colorsReshaped, cv2.ROTATE_90_CLOCKWISE)
        destaggered = client.destagger(metadataLidar, rotated, inverse=True)

        width = destaggered.shape[1] # keep original width
        height = 64*4
        dim = (width, height)

        rgbTaller = cv2.resize(destaggered, dim, interpolation = cv2.INTER_NEAREST)
        #rgbTaller = cv2.cvtColor(rgbSmaller.astype('float32'), cv2.COLOR_BGR2RGB)

        #cv2.imwrite(f"/home/potetsos/lagrinatorn/master/rellisOutput/labels/{drive}/{filename[12:-4]}.png", rgbTaller)
        #np.save(f"/home/potetsos/lagrinatorn/master/rellisOutput/numpyPointcloud/{drive}/{filename[12:-4]}.npy", points)
        #np.save(f"/home/potetsos/lagrinatorn/master/rellisOutput/numpyColorsCloud/{drive}/{filename[12:-4]}.npy", colors)

        
    
