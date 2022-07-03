from inferanceLidar import Inferance
from os import listdir
from tqdm import tqdm
from detectron2.data.detection_utils import read_image
import time
import cv2
import numpy as np
import os

inferance = Inferance(loggingFolder="lidarVanlig", modelName="ffrLidarFinal")

#outputFolder = "/lhome/asbjotof/asbjotof/2022/outputImages/lidarWithCounterClassImage"
outputFolders = "/lhome/asbjotof/master/outputImaes/lidarVanlig"
#inputfoler = "datasetLidarTilNo/train/images"
#inputfoler = "datasetLidarTilNo/train/imagesNumpy"
#inputfoler = "inputImages/lidarDatasetNumpy"
#inputfoler = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdataset/test/numpyImage"
#inputfoler = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdatasetWithRGBCounter/test/lidarNumpyImage"

#inputfoler = "ffiLiDARdatasetWithRGBCounter/test/lidarNumpyImage"
inputfolersPath = "/lhome/asbjotof/asbjotof/2022/data/structuredInFolders"





inputfolers = listdir(inputfolersPath)

inputfolers.sort()
        
for inputfoler in [inputfolers[5]]:
    print(inputfoler)
    inputfolerFullPath = f"{inputfolersPath}/{inputfoler}"
    #inputfolerFullPath = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiRGBdataset3/test/image"
    outputFolder = f"{outputFolders}/{inputfoler}"
    #outputFolder = "/lhome/asbjotof/asbjotof/2022/bilder/outputTest3"
    files = listdir(inputfolerFullPath)

    files.sort()
    #os.makedirs(outputFolder, exist_ok=True)

    os.makedirs(f"{outputFolder}/all", exist_ok=True)
    os.makedirs(f"{outputFolder}/pred", exist_ok=True)
            
    #for fileName in tqdm(files):


    for fileName in tqdm(files):
        #frame = read_image(inputfoler + "/" +fileName, format="BGR")
        #try:
        #print(fileName)
        #numpyPath = '/'.join(filePaths)[:-4] + ".npy"
        
        numpyPath = f"{inputfolerFullPath}/{fileName}"
        numpyImage = np.load(numpyPath)
        #startTime = time.time()
        vis_panoptic, rgb_img, classImage = inferance.segmentImage(numpyImage, fileName)
        #print(rgb_img)
        #print(rgb_img.shape)
        #print(type(classImage))
        predClassNumpy = np.array(classImage, dtype=np.int)
        #print(f"{outputFolder}/{fileName[-4]}.png")
        #cv2.imwrite(f"{outputFolder}/{fileName[:-4]}.png", vis_panoptic)
        combinedFrame = np.vstack((numpyImage[:,:,0:3], vis_panoptic))
        cv2.imwrite(f"{outputFolder}/all/{fileName[:-4]}.png",combinedFrame)
        cv2.imwrite(f"{outputFolder}/pred/{fileName[:-4]}.png",rgb_img)
        #diffTime = time.time() - startTime
        #print("segmentering:", diffTime)
        #except:
        #    print(fileName)
        #    numpyImage = np.load(numpyPath)
        #    print(numpyImage.shape)
#inferance.writeDatasetToFile()