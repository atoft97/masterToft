import cv2
from os import listdir, rename, makedirs
from tqdm import tqdm
import numpy as np

loadPath = "/home/potetsos/lagrinatorn/master/rellisOutput/destagg"
drivePaths = listdir(loadPath)

for drivePath in drivePaths:
    fullDrivePath = loadPath + "/" + drivePath 
    signalPath = fullDrivePath + "/new_signal_image" #choose singal image because it has the same number of images as the other folders that are going to be combined

    imageNames = listdir(signalPath)
    imageNames.sort()

    print("Drive: ", drivePath)
    makedirs("/home/potetsos/lagrinatorn/master/rellisOutput/combinedImages/" + drivePath, exist_ok=True)
    makedirs("/home/potetsos/lagrinatorn/master/rellisOutput/combinedNumpy/" + drivePath, exist_ok=True)

    for i, imageName in enumerate(tqdm((imageNames))):
        #try:
        img1 = cv2.imread(fullDrivePath + "/new_signal_image/" + imageName, 0)
        img2 = cv2.imread(fullDrivePath + "/new_reflec_image/" + imageName, 0)
        img3 = cv2.imread(fullDrivePath + "/new_range_bright/" + imageName, 0)
        img4 = cv2.imread(fullDrivePath + "/new_nearir_image/" + imageName, 0)
        img = cv2.merge((img1, img2, img3))
        cv2.imwrite("/home/potetsos/lagrinatorn/master/rellisOutput/combinedImages/" + drivePath + "/" + imageName, img)

        stacked4channels = np.dstack((img1, img2, img3, img4))
        np.save("/home/potetsos/lagrinatorn/master/rellisOutput/combinedNumpy/" + drivePath + "/" + imageName[:-4] + ".npy", stacked4channels)
        #except:
        #    print("failed on: ", fullDrivePath, " Image number:", imageName)

        #break

