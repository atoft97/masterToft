import open3d as o3d
import numpy as np
import cv2
from ouster import client
import os
from tqdm import tqdm

metadataPath = "lidar_metadata.json"
with open(metadataPath, "r") as f:
    metadataLidar = client.SensorInfo(f.read())


pointFullPath = f"/home/potetsos/lagrinatorn/master/pointCloudAll/stand_still_short00001.npy"
colorPath = f"/home/potetsos/skule/2022/masterCode/masterToft/data/combinedImages/stand_still_short/stand_still_short00001.png"
pointData = np.load(pointFullPath)

colorImage = cv2.imread(colorPath)




dim = (2048, 64)
colorSmaller = cv2.resize(colorImage, dim, interpolation = cv2.INTER_NEAREST)
colorSmaller = cv2.cvtColor(colorSmaller.astype("float32"), cv2.COLOR_BGR2RGB)
colorStaggered = client.destagger(metadataLidar, colorSmaller, inverse=True)
colorReshaped = (colorStaggered.astype(float) / 255).reshape(-1, 3)
print(colorSmaller.shape)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointData)
pcd.colors = o3d.utility.Vector3dVector(colorReshaped)

o3d.visualization.draw_geometries([pcd])