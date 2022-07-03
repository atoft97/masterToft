import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import matplotlib.image as mpimg
import yaml
import open3d as o3d
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
from ouster import client
from skimage.segmentation import watershed, expand_labels
import os

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:,:3]

def print_projection_plt(points, color, image, lidarImageColor, lidarLabelColor):
    #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #print(image.shape)
    #print("hmm", lidarLabelColor.shape)
    image3Channels = np.zeros(image.shape)
    image1Channel = np.zeros(image.shape)
    image4Channel = np.zeros((image.shape[0], image.shape[1], image.shape[2]+1))
    imageLabel = np.zeros(image.shape)
    #print(image4Channel.shape)
    #image = np.zeros(image.shape)

    #print("1", points.shape[1])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        pixelColor = (int(lidarImageColor[i][0]), int(lidarImageColor[i][1]),int(lidarImageColor[i][2]))
        cv2.circle(image, (np.int32(points[0][i]),np.int32(points[1][i])),7, pixelColor,-1)

        cv2.circle(image3Channels, (np.int32(points[0][i]),np.int32(points[1][i])),7, pixelColor,-1)

        pixelColor = (int(lidarImageColor[i][3]), int(lidarImageColor[i][3]),int(lidarImageColor[i][3]))
        cv2.circle(image1Channel, (np.int32(points[0][i]),np.int32(points[1][i])),7, pixelColor,-1)

        pixelColor = (int(lidarLabelColor[i][0]), int(lidarLabelColor[i][1]),int(lidarLabelColor[i][2]))
        cv2.circle(imageLabel, (np.int32(points[0][i]),np.int32(points[1][i])),7, pixelColor,-1)

        cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)

    image4Channel[:,:,0:3] = image3Channels
    image4Channel[:,:,3] = image1Channel[:,:,0]

    #print(image4Channel[1000:1100])
    #expanded = expand_labels(image, distance=3)
        #break

    #return(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB),image4Channel[:,:,0:3], image4Channel, imageLabel)
    return(image,image4Channel[:,:,0:3], image4Channel, imageLabel)
    #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def depth_color(val, min_d=0, max_d=120):
    np.clip(val, 0, max_d, out=val) 
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def points_filter(points,img_width,img_height,P,RT, lidarImage, lidarLabelImage):
    #print(points.shape)
    #print(img_width)
    #print(img_height)
    #print(P)
    #print(RT)

    ctl = RT
    ctl = np.array(ctl)
    fov_x = 2*np.arctan2(img_width, 2*P[0,0])*180/3.1415926+10
    fov_y = 2*np.arctan2(img_height, 2*P[1,1])*180/3.1415926+10
    R= np.eye(4)
    p_l = np.ones((points.shape[0],points.shape[1]+1))
    p_l[:,:3] = points


    p_l_lidarImage = np.ones((lidarImage.shape[0],lidarImage.shape[1]+1))
    p_l_lidarImage[:,:4] = lidarImage

    p_l_lidarLabelImage = np.ones((lidarLabelImage.shape[0],lidarLabelImage.shape[1]+1))
    p_l_lidarLabelImage[:,:3] = lidarLabelImage
    #print("usikker", p_l_lidarImage.shape)



    p_c = np.matmul(ctl,p_l.T)
    #print(ctl)
    p_c = p_c.T
    x = p_c[:,0]
    y = p_c[:,1]
    z = p_c[:,2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    xangle = np.arctan2(x, z)*180/np.pi;
    yangle = np.arctan2(y, z)*180/np.pi;
    flag2 = (xangle > -fov_x/2) & (xangle < fov_x/2)
    flag3 = (yangle > -fov_y/2) & (yangle < fov_y/2)
    res = p_l[flag2&flag3,:3]
    res = np.array(res)



    resLidarImage = p_l_lidarImage[flag2&flag3,:4]
    resLidarImage = np.array(resLidarImage)
    #print("sikker", resLidarImage.shape)

    resLidarLabelImage = p_l_lidarLabelImage[flag2&flag3,:4]
    resLidarLabelImage = np.array(resLidarLabelImage)


    x = res[:, 0]
    y = res[:, 1]
    z = res[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    color = depth_color(dist, 0, 20)
    return res,color,resLidarImage, resLidarLabelImage

def get_cam_mtx(filepath):
    data = np.loadtxt(filepath)
    P = np.zeros((3,3))
    P[0,0] = data[0]
    P[1,1] = data[1]
    P[2,2] = 1
    P[0,2] = data[2]
    P[1,2] = data[3]
    return P

def get_mtx_from_yaml(filepath,key='os1_cloud_node-pylon_camera_node'):
    with open(filepath,'r') as f:
        data = yaml.load(f,Loader= yaml.Loader)
    q = data[key]['q']
    q = np.array([q['x'],q['y'],q['z'],q['w']])
    t = data[key]['t']
    t = np.array([t['x'],t['y'],t['z']])
    R_vc = Rotation.from_quat(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4,4)
    RT[:3,:3] = R_vc
    RT[:3,-1] = t
    RT = np.linalg.inv(RT)
    return RT


distCoeff = np.array([-0.134313,-0.025905,0.002181,0.00084,0])
distCoeff = distCoeff.reshape((5,1))

P = get_cam_mtx('camera_info_rellis.txt')
#print(P)

RT= get_mtx_from_yaml('transformsrellis.yaml')

#image = cv2.imread('frame000104-1581624663_149.jpg')
#image = cv2.imread('/home/potetsos/lagrinatorn/master/rellisOutput/rgb/train/image/1581624110_250.jpg')
outputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/projectedDataset"



os.makedirs(f"{outputPath}/train/visualize", exist_ok=True)
os.makedirs(f"{outputPath}/val/visualize", exist_ok=True)
os.makedirs(f"{outputPath}/test/visualize", exist_ok=True)

os.makedirs(f"{outputPath}/train/lidarImage", exist_ok=True)
os.makedirs(f"{outputPath}/val/lidarImage", exist_ok=True)
os.makedirs(f"{outputPath}/test/lidarImage", exist_ok=True)

os.makedirs(f"{outputPath}/train/lidarNumpy", exist_ok=True)
os.makedirs(f"{outputPath}/val/lidarNumpy", exist_ok=True)
os.makedirs(f"{outputPath}/test/lidarNumpy", exist_ok=True)

os.makedirs(f"{outputPath}/train/lidarLabels", exist_ok=True)
os.makedirs(f"{outputPath}/val/lidarLabels", exist_ok=True)
os.makedirs(f"{outputPath}/test/lidarLabels", exist_ok=True)

metadata_path = "lidar_metadata.json"
with open(metadata_path, 'r') as f:
    metadataLidar = client.SensorInfo(f.read())

def lidarRGBcombine(lidarPath, rgbPath, lidarImagePath, lidarNumpyPath, lidarLabelPath, trainValTest, filename):
    image = cv2.imread(rgbPath)
    #print("hei")
    #cv2.imwrite("inputImage.png", image)

    lidarImage = cv2.imread(lidarImagePath)
    lidarLabelImage = cv2.imread(lidarLabelPath)
    #print("lidar image", lidarImage.shape)
    lidarNumpyImage = np.load(lidarNumpyPath)
    #print("lidar numpy image", lidarNumpyImage.shape)
    reshapedLidarImage = lidarImage.reshape((2048*64, -1))
    #print("lidar image", reshapedLidarImage.shape)
    reshapedLidarNumpyImage = lidarNumpyImage.reshape((2048*64, -1))
    #print("lidar numpy image", reshapedLidarNumpyImage.shape)
    #pcd2 = o3d.io.read_point_cloud('/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/stand_still_short/stand_still_short00000.npy')
    #points2 = np.asarray(pcd2.points)
    points = np.load(lidarPath)
    #print(points.shape)

    width = 2048
    height = 64
    dim = (width, height)




    rgbSmaller = cv2.resize(lidarImage, dim, interpolation = cv2.INTER_NEAREST)
    #rgbSmaller = cv2.cvtColor(rgbSmaller.astype('float32'), cv2.COLOR_BGR2RGB)
    cv2.imwrite("utenStagg.png", rgbSmaller)
    rgbSmaller = rgbSmaller.astype('float32')
    segmentation_img_staggered = client.destagger(metadataLidar, rgbSmaller, inverse=True)
    cv2.imwrite("stagg.png", segmentation_img_staggered)
    segmentation_img_staggeredFloat = (segmentation_img_staggered.astype(float)).reshape(-1, 3)
    #print(segmentation_img_staggeredFloat.shape)

    lidarLabelImageSmaller = cv2.resize(lidarLabelImage, dim, interpolation = cv2.INTER_NEAREST)
    lidarLabelImageSmaller = lidarLabelImageSmaller.astype('float32')
    lidarLabelImageSmaller_staggered = client.destagger(metadataLidar, lidarLabelImageSmaller, inverse=False)
    lidarLabelImageSmaller_staggered = cv2.rotate(lidarLabelImageSmaller_staggered, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lidarLabelImageSmaller_staggeredFloat = (lidarLabelImageSmaller_staggered.astype(float)).reshape(-1, 3)


    lidarNumpyImageSmaller = cv2.resize(lidarNumpyImage, dim, interpolation = cv2.INTER_NEAREST)
    lidarNumpyImageSmaller = lidarNumpyImageSmaller.astype('float32')
    lidarNumpyImageSmaller_staggered = client.destagger(metadataLidar, lidarNumpyImageSmaller, inverse=False)
    lidarNumpyImageSmaller_staggered = cv2.rotate(lidarNumpyImageSmaller_staggered, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #cv2.imwrite("stagg.png", segmentation_img_staggered)
    lidarNumpyImageSmaller_staggeredFloat = (lidarNumpyImageSmaller_staggered.astype(float)).reshape(-1, 4)
    #pcd = o3d.io.read_point_cloud('/home/potetsos/lagrinatorn/master/Rellis-3D/00003/os1_cloud_node_color_ply/frame000349-1581624110_185.ply')
    #points = np.asarray(pcd.points)
    #print(points.shape)

    #points = load_from_bin('000104.bin')
    #print(points.shape)
    #image = cv2.imread('frame000104-1581624663_149.jpg')

    img_height, img_width, channels = image.shape

    #print(RT)
    P = get_cam_mtx('camera_info_rellis.txt')
    RT= get_mtx_from_yaml('transformsrellis.yaml')
    #print(P)
    #print(RT)
    R_vc = RT[:3,:3]
    T_vc = RT[:3,3]
    T_vc = T_vc.reshape(3, 1)
    rvec,_ = cv2.Rodrigues(R_vc)
    tvec = T_vc
    xyz_v, c_, lidarImageColor, lidarLabelColor = points_filter(points,img_width,img_height,P,RT, lidarNumpyImageSmaller_staggeredFloat, lidarLabelImageSmaller_staggeredFloat)
    #print("ting", xyz_v.shape)
    #print("color", c_.shape)
    #print("lidarImageColor", lidarImageColor)
    #print("lidarImageColor", lidarLabelColor)
    #print(xyz_v.shape)


    imgpoints, _ = cv2.projectPoints(xyz_v[:,:],rvec, tvec, P, distCoeff)
    #print("sluttpunkt", imgpoints.shape)
    imgpoints = np.squeeze(imgpoints,1)
    imgpoints = imgpoints.T
    visualised, lidarImage, lidarNumpy, lidarLabels = print_projection_plt(points=imgpoints, color=c_, image=image, lidarImageColor=lidarImageColor, lidarLabelColor=lidarLabelColor)

    #visualize, lidarImage, lidarNumpy, lidarLabels 
    cv2.imwrite(f"{outputPath}/{trainValTest}/visualize/{filename}.png", visualised)
    cv2.imwrite(f"{outputPath}/{trainValTest}/lidarImage/{filename}.png", lidarImage)
    cv2.imwrite(f"{outputPath}/{trainValTest}/lidarNumpy/{filename}.png", lidarNumpy)
    #np.save(f"{outputPath}/{trainValTest}/lidarNumpy/{filename}.npy", lidarNumpy)
    cv2.imwrite(f"{outputPath}/{trainValTest}/lidarLabels/{filename}.png", lidarLabels)
    #plt.subplots(1,1, figsize = (20,20) )
    #plt.title("Velodyne points to camera image Result")
    #plt.imshow(visualised)
    #plt.show()

#Cc
#os_sensor

#lidarRGBcombine('/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/stand_still_short/stand_still_short00000.npy', '/home/potetsos/lagrinatorn/master/bilder/rgb/stand_still_short/stand_still_short_00000.png', "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedImages/stand_still_short/stand_still_short00001.png", "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/numpyImage/lidar/4.npy", "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/colorSegment/lidar/4.png")