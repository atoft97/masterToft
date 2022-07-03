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
import copy

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:,:3]

def print_projection_plt(points, color, image, lidarImageColor, lidarLabelColor, resLidarIndex):
    #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #print(image.shape)
    #print("hmm", lidarLabelColor.shape)
    image3Channels = np.zeros(image.shape)
    image1Channel = np.zeros(image.shape)
    image4Channel = np.zeros((image.shape[0], image.shape[1], image.shape[2]+1))
    imageLabel = np.zeros(image.shape)
    #print(image4Channel.shape)
    #image = np.zeros(image.shape)

    width = 2048
    height = 64
    dim = (width, height)
    rgbToLidarImage = np.zeros(dim)
    #pointsColors = np.zeros((points.shape[1],3))
    #print(pointsColors.shape)

    pointsColors = []
    resLidarIndexNew = []

    imageCopy = copy.copy(image)

    for i in range(points.shape[1]):


        #print(0 < pointsX < imageShapeX)

        #print((0 < pointsX < imageShapeX) and (0 < pointsY < imageShapeY))

        #print("\n")
        imageShapeX = image.shape[0]
        imageShapeY = image.shape[1]
        pointsX = int(points[1][i])
        pointsY = int(points[0][i])

        if((0 < pointsX < imageShapeX) and (0 < pointsY < imageShapeY)):


            pixelColor = (int(lidarImageColor[i][0]), int(lidarImageColor[i][1]),int(lidarImageColor[i][2]))
            cv2.circle(imageCopy, (np.int32(points[0][i]),np.int32(points[1][i])),3, pixelColor,-1)

            #cv2.circle(image3Channels, (np.int32(points[0][i]),np.int32(points[1][i])),5, pixelColor,-1)

            pixelColor = (int(lidarImageColor[i][3]), int(lidarImageColor[i][3]),int(lidarImageColor[i][3]))
            cv2.circle(image1Channel, (np.int32(points[0][i]),np.int32(points[1][i])),5, pixelColor,-1)

            pixelColor = (int(lidarLabelColor[i][0]), int(lidarLabelColor[i][1]),int(lidarLabelColor[i][2]))
            cv2.circle(imageLabel, (np.int32(points[0][i]),np.int32(points[1][i])),5, pixelColor,-1)




            rgbColor = image[(np.int32(points[1][i]),np.int32(points[0][i]))]
            #rgbToLidarImage[]
            #print(rgbColor)
            #print(pixelColor)
            #pixelColor = (int(lidarImageColor[i][0]), int(lidarImageColor[i][1]),int(lidarImageColor[i][2]))
            #pointsColors[i,:] = pixelColor
            #pointsColors[i,:] = rgbColor
            pointsColors.append(rgbColor)
            resLidarIndexNew.append(resLidarIndex[i])


            #fancyColor = (rgbColor[0], rgbColor[1], rgbColor[2])
            #print(fancyColor)

            #cv2.circle(image3Channels, (np.int32(points[0][i]),np.int32(points[1][i])),5, rgbColor,-1)
        
        

    image4Channel[:,:,0:3] = image3Channels
    image4Channel[:,:,3] = image1Channel[:,:,0]

    #print(image4Channel[1000:1100])

    #print(image4Channel[1000:1100])
    #expanded = expand_labels(image, distance=3)
        #break
    resLidarIndexNewNP = np.array(resLidarIndexNew)
    #print("for")
    pointsColors = np.array(pointsColors)
    #print("etter")
    #print(pointsColors)
    return(imageCopy,image4Channel[:,:,0:3], image4Channel, imageLabel, pointsColors, resLidarIndexNewNP)
    #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def depth_color(val, min_d=0, max_d=120):
    np.clip(val, 0, max_d, out=val) 
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def points_filter(points,img_width,img_height,P,RT, lidarImage, lidarLabelImage, lidarIndex):
    ctl = RT
    ctl = np.array(ctl)
    fov_x = 2*np.arctan2(img_width, 2*P[0,0])*180/3.1415926+10
    fov_y = 2*np.arctan2(img_height, 2*P[1,1])*180/3.1415926+10
    R= np.eye(4)
    #print("weird", points.shape)
    p_l = np.ones((points.shape[0],points.shape[1]+1))
    p_l[:,:3] = points


    p_l_lidarImage = np.ones((lidarImage.shape[0],lidarImage.shape[1]+1))
    p_l_lidarImage[:,:4] = lidarImage

    p_l_lidarLabelImage = np.ones((lidarLabelImage.shape[0],lidarLabelImage.shape[1]+1))
    p_l_lidarLabelImage[:,:3] = lidarLabelImage

    p_l_lidarIndex = np.ones((lidarIndex.shape[0]))
    p_l_lidarIndex = lidarIndex
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

    resLidarLabelImage = p_l_lidarLabelImage[flag2&flag3,:3]
    resLidarLabelImage = np.array(resLidarLabelImage)

    resLidarIndex = p_l_lidarIndex[flag2&flag3]
    resLidarIndex = np.array(resLidarIndex)


    x = res[:, 0]
    y = res[:, 1]
    z = res[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    color = depth_color(dist, 0, 20)
    return res,color,resLidarImage, resLidarLabelImage, resLidarIndex

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

    euler = euler_from_quaternion(q)

    #print(euler)
    xdeg = math.degrees(euler[0])+6.5
    ydeg = math.degrees(euler[1])-1
    zdeg = math.degrees(euler[2])-16.4
    #print(xdeg, ydeg, zdeg)

    quaternion = quaternion_from_euler(math.radians(xdeg),math.radians(ydeg), math.radians(zdeg))
    q = quaternion

    t = data[key]['t']
    t = np.array([t['x'],t['y'],t['z']])
    R_vc = Rotation.from_quat(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4,4)
    RT[:3,:3] = R_vc
    RT[:3,-1] = t
    RT = np.linalg.inv(RT)
    return RT


distCoeff = np.array([-0.006029910205705422, 0.014249757232347872, 0.0008574690513132267, 0.0025905772428531488, 0])
distCoeff = distCoeff.reshape((5,1))

P = get_cam_mtx('camera_info.txt')
#print(P)

RT= get_mtx_from_yaml('transforms.yaml')

#image = cv2.imread('frame000104-1581624663_149.jpg')
#image = cv2.imread('/home/potetsos/lagrinatorn/master/rellisOutput/rgb/train/image/1581624110_250.jpg')

metadata_path = "lidar_metadata.json"
with open(metadata_path, 'r') as f:
    metadataLidar = client.SensorInfo(f.read())

def lidarRGBcombine(lidarPath, rgbPath, lidarImagePath, lidarNumpyPath, lidarLabelPath, lidarLabelIdPath):
    image = cv2.imread(rgbPath)

    lidarImage = cv2.imread(lidarImagePath)
    lidarLabelImage = cv2.imread(lidarLabelPath)
    lidarLabelIDImage = cv2.imread(lidarLabelIdPath, -1)
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
    rgbSmaller = rgbSmaller.astype('float32')
    segmentation_img_staggered = client.destagger(metadataLidar, rgbSmaller, inverse=True)
    cv2.imwrite("stagg.png", segmentation_img_staggered)
    segmentation_img_staggeredFloat = (segmentation_img_staggered.astype(float)).reshape(-1, 3)
    #print(segmentation_img_staggeredFloat.shape)

    lidarLabelImageSmaller = cv2.resize(lidarLabelImage, dim, interpolation = cv2.INTER_NEAREST)
    lidarLabelImageSmaller = lidarLabelImageSmaller.astype('float32')
    #print("init shape", lidarLabelImageSmaller.shape)
    lidarLabelImageSmaller_staggered = client.destagger(metadataLidar, lidarLabelImageSmaller, inverse=True)
    lidarLabelImageSmaller_staggeredFloat = (lidarLabelImageSmaller_staggered.astype(float)).reshape(-1, 3)




    lidarLabelImageIDSmaller = cv2.resize(lidarLabelIDImage, dim, interpolation = cv2.INTER_NEAREST)
    lidarLabelImageIDSmaller = lidarLabelImageIDSmaller.astype('float32')
    #print("init shape", lidarLabelImageSmaller.shape)
    lidarLabelImageIDSmaller_staggered = client.destagger(metadataLidar, lidarLabelImageIDSmaller, inverse=True)
    lidarLabelImageIDSmaller_staggeredFloat = (lidarLabelImageIDSmaller_staggered.astype(float)).reshape(-1)



    
    lidarIndex = np.arange(64*2048)
    
    #lidarIndex = cv2.resize(lidarLabelImage, dim, interpolation = cv2.INTER_NEAREST)
    #lidarIndex_destaggered = client.destagger(metadataLidar, lidarLabelImageSmaller, inverse=True)

    lidarNumpyImageSmaller = cv2.resize(lidarNumpyImage, dim, interpolation = cv2.INTER_NEAREST)
    lidarNumpyImageSmaller = lidarNumpyImageSmaller.astype('float32')
    lidarNumpyImageSmaller_staggered = client.destagger(metadataLidar, lidarNumpyImageSmaller, inverse=True)
    #cv2.imwrite("stagg.png", segmentation_img_staggered)
    lidarNumpyImageSmaller_staggeredFloat = (lidarNumpyImageSmaller_staggered.astype(float)).reshape(-1, 4)
    #pcd = o3d.io.read_point_cloud('/home/potetsos/lagrinatorn/master/Rellis-3D/00003/os1_cloud_node_color_ply/frame000349-1581624110_185.ply')
    #points = np.asarray(pcd.points)
    #print(points.shape)

    #points = load_from_bin('000104.bin')
    #print(points.shape)

    img_height, img_width, channels = image.shape

    #print(RT)

    R_vc = RT[:3,:3]
    T_vc = RT[:3,3]
    T_vc = T_vc.reshape(3, 1)
    rvec,_ = cv2.Rodrigues(R_vc)
    tvec = T_vc
    #print("for", lidarLabelImageSmaller_staggeredFloat.shape)
    xyz_v, c_, lidarImageColor, lidarLabelColor, resLidarIndex = points_filter(points,img_width,img_height,P,RT, lidarNumpyImageSmaller_staggeredFloat, lidarLabelImageSmaller_staggeredFloat, lidarIndex)
    #print("ting", xyz_v.shape)
    #print("color", c_.shape)
    #print("lidarImageColor", lidarImageColor.shape)
    #print("lidarLabelColor", lidarLabelColor.shape)
    #print("lidarIndex", resLidarIndex.shape)
    #print(resLidarIndex)
    #print(xyz_v.shape)


    imgpoints, _ = cv2.projectPoints(xyz_v[:,:],rvec, tvec, P, distCoeff)
    #print("sluttpunkt", imgpoints.shape)
    imgpoints = np.squeeze(imgpoints,1)
    imgpoints = imgpoints.T
    visualised, lidarImage, lidarNumpy, lidarLabels, pointsColors, resLidarIndexNew = print_projection_plt(points=imgpoints, color=c_, image=image, lidarImageColor=lidarImageColor, lidarLabelColor=lidarLabelColor, resLidarIndex=resLidarIndex)

    #cv2.imwrite("projektertLidar.png", visualised)

    #plt.subplots(1,1, figsize = (20,20))
    #plt.title("Velodyne points to camera image Result")
    #plt.imshow(visualised)

    #print(pointsColors.shape)
    #print(pointsColors[100:200,:])
    #print("lidar form", lidarNumpy.shape)
    

    colorImageBack = np.zeros((64*2048, 3))
    lidarNumpyBack = np.zeros((64*2048, 4))
    lidarLabelBack = np.zeros((64*2048, 3))
    lidarLabelIdBack = np.zeros((64*2048))
    #print("colros hspae", pointsColors.shape)
    #print("new shape", resLidarIndexNew.shape)
    for i in range(len(resLidarIndexNew)):
        color = pointsColors[i]
        #numpy4channelColor = lidarNumpyImageSmaller_staggeredFloat[i]
        index = resLidarIndexNew[i]

        colorImageBack[index] = color

        #numpy4channelColor = 
        lidarNumpyBack[index] = lidarNumpyImageSmaller_staggeredFloat[index]

        lidarLabelBack[index] = lidarLabelImageSmaller_staggeredFloat[index]

        lidarLabelIdBack[index] = lidarLabelImageIDSmaller_staggeredFloat[index]

        #lidarNumpyBack[index] = numpy4channelColor
    #for i in range(len(resLidarIndex)):
    #    index = resLidarIndex[i]
    #    numpy4channelColor = lidarNumpyImageSmaller_staggeredFloat[index]


        #print(color)
    colorImageBackReshaped = colorImageBack.reshape(64, 2048, 3)
    lidarNumpyBackReshaped = lidarNumpyBack.reshape(64, 2048, 4)
    lidarLabelBackReshaped = lidarLabelBack.reshape(64, 2048, 3)
    lidarLabelIDBackReshaped = lidarLabelIdBack.reshape(64, 2048)
    #print(colorImageBack.shape)
    #print(resLidarIndex.shape)

    colorImageBackDestaggered = client.destagger(metadataLidar, colorImageBackReshaped, inverse=False)
    lidarNumpyBackReshaped = client.destagger(metadataLidar, lidarNumpyBackReshaped, inverse=False)
    lidarLabelBackReshaped = client.destagger(metadataLidar, lidarLabelBackReshaped, inverse=False)
    lidarLabelIDBackReshaped = client.destagger(metadataLidar, lidarLabelIDBackReshaped, inverse=False)

    cv2.imwrite("tmp/pointColor.png", colorImageBackDestaggered)
    cv2.imwrite("tmp/lidarBack.png", lidarNumpyBackReshaped[:,:,0:3])
    cv2.imwrite("tmp/labelBack.png", lidarLabelBackReshaped)

    return(colorImageBackDestaggered, lidarNumpyBackReshaped, lidarLabelBackReshaped, lidarLabelIDBackReshaped, visualised)

    #print(lidarLabels)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_v)
    #print(xyz_v)
    #print(c_)
    pcd.colors = o3d.utility.Vector3dVector(lidarLabelColor)

    #return

    #o3d.visualization.draw_geometries([pcd])
    #plt.show()

#Cc
#os_sensor

#lidarRGBcombine('/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/stand_still_short/stand_still_short00000.npy', '/home/potetsos/lagrinatorn/master/bilder/rgb/stand_still_short/stand_still_short_00000.png', "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedImages/stand_still_short/stand_still_short00001.png", "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/numpyImage/lidar/4.npy", "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/colorSegment/lidar/4.png")

#rgbFullPath = '/home/potetsos/lagrinatorn/master/bilder/rgb/stand_still_short/stand_still_short_00000.png'
lidarFullPath = "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedImages/stand_still_short/stand_still_short00001.png"
lidarNumpyFullPath = '/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/stand_still_short/stand_still_short00000.npy'
lidarLabelFullPath = "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/colorSegment/lidar/4.png"
lidarCloudFullPath = "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/numpyImage/lidar/4.npy"

rgbFullPath = '/home/potetsos/lagrinatorn/master/ffiRGBdatasetAndLidar/train/image/plainsDrive_frame01947.png'
lidarFullPath = "/home/potetsos/lagrinatorn/master/ffiRGBdatasetAndLidar/train/lidarImage/plains_drive04550.png"
lidarNumpyFullPath = '/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/plains_drive/plains_drive04550.npy'
lidarLabelFullPath = "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/colorSegment/lidar/4.png"
lidarCloudFullPath = "/home/potetsos/lagrinatorn/master/ffiRGBdatasetAndLidar/train/lidarNumpy/plains_drive04550.npy"


#lidarRGBcombine(lidarNumpyFullPath,rgbFullPath , lidarFullPath,lidarCloudFullPath , lidarLabelFullPath)