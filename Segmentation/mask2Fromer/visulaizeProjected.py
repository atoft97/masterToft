import open3d as o3d
import cv2
import numpy as np
from ouster import client

metadata_path = "lidar_metadata.json"
with open(metadata_path, 'r') as f:
    metadataLidar = client.SensorInfo(f.read())

colorsPath = "/lhome/asbjotof/asbjotof/2022/lidarFromRGBSegmented/pred/75.png"
colorsImage = cv2.imread(colorsPath)

pointCloudPath = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdatasetWithRGBCounter/test/lidarPointCloud/75.npy"
pointCloudPoints = np.load(pointCloudPath)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointCloudPoints)
#print(xyz_v)
#print(c_)


dim = (2048, 64)
labelSmaller = cv2.resize(colorsImage, dim, interpolation = cv2.INTER_NEAREST)
labelSmaller = cv2.cvtColor(labelSmaller.astype("float32"), cv2.COLOR_BGR2RGB)
labelStaggered = client.destagger(metadataLidar, labelSmaller, inverse=True)
labelReshaped = (labelStaggered.astype(float) / 255).reshape(-1, 3)

pcd.colors = o3d.utility.Vector3dVector(labelReshaped)



o3d.visualization.draw_geometries([pcd])