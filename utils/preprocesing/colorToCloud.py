import open3d as o3d
import numpy as np
import cv2
from ouster import client

metadataPath = "lidar_metadata.json"
with open(metadataPath, "r") as f:
    metadataLidar = client.SensorInfo(f.read())

pointPath = "/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/plains_drive/plains_drive04579.npy"
lablePath = "/home/potetsos/lagrinatorn/master/dataFromCvat/SegmentationClass/plains_drive04579.png"
pointData = np.load(pointPath)

labelImage = cv2.imread(lablePath)
dim = (2048, 64)
labelSmaller = cv2.resize(labelImage, dim, interpolation = cv2.INTER_NEAREST)
labelSmaller = cv2.cvtColor(labelSmaller.astype("float32"), cv2.COLOR_BGR2RGB)
labelStaggered = client.destagger(metadataLidar, labelSmaller, inverse=True)
labelReshaped = (labelStaggered.astype(float) / 255).reshape(-1, 3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointData)
pcd.colors = o3d.utility.Vector3dVector(labelReshaped)

o3d.visualization.draw_geometries([pcd])