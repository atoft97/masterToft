from inferance import Inferance
#from inferanceRellisRGB import Inferance
#from inferanceRellisLidar import InferanceRellisLidar
from os import listdir, makedirs
from tqdm import tqdm
from detectron2.data.detection_utils import read_image
import time
import cv2
import numpy as np
#inferance = Inferance(loggingFolder="slettDenne", modelName="rellisBRG1t")
#inferance = Inferance(loggingFolder="test2", modelName="ffiRGButenOther")
#inferance = Inferance(loggingFolder="test3", modelName="3channelLidar")
#inferance = Inferance(loggingFolder="test3", modelName="swinLarge")
inferance = Inferance(loggingFolder="test3", modelName="ffiRGBLast")
#swinLarge

#inferance = Inferance(loggingFolder="test2", modelName="ffiRGBfinal")


#inferance = InferanceRellisLidar(loggingFolder="slettDenneLidar", modelName="rellisLidar")


#outputFolder = "outputImages/Rellis/lidar"
outputFolders = "/lhome/asbjotof/asbjotof/2022/segmentedRGBOnly"
#inputfoler = "../../maskFormer/inferance/inputImages2"
#inputfoler = "rgb/test/image"
innputFoldersPath = "/lhome/asbjotof/master/bilder/rgb"
inputfolers = listdir(innputFoldersPath)
inputfolers.sort()
#inputfoler = "lidarDataset/test/numpyImage"
#inputfolers = ["test"]
#print(inputfolers[9:10])
#print(plain)
for inputfoler in inputfolers[9:10]:
    print(inputfoler)
    inputfolerFullPath = f"{innputFoldersPath}/{inputfoler}"
    #inputfolerFullPath = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiRGBdataset3/test/image"
    #inputfolerFullPath = "ffiLiDARdataset/test/3channelImage"
    
    outputFolder = f"{outputFolders}/{inputfoler}"
    #outputFolder = "/lhome/asbjotof/asbjotof/2022/outputImages"
    #outputFolder = "/lhome/asbjotof/asbjotof/2022/outputImages/lidar3channel"
    files = listdir(inputfolerFullPath)

    files.sort()
    makedirs(outputFolder, exist_ok=True)
            
    for fileName in tqdm(files[1500:]):
        #frame = read_image("/lhome/asbjotof/asbjotof/2022/bilder/rgb/stand_still_short/stand_still_short_00000.png", format="BGR")
        frame = read_image(inputfolerFullPath + "/" +fileName, format="BGR")

        #frame = read_image("ffiRGBdataset/train/image/2022-04-08-13-49-07_00400.png", format="BGR")

        #frame = np.load(f"{inputfoler}/{fileName}")
        #startTime = time.time()
        vis_panoptic, rgb_img, classImage = inferance.segmentImage(frame, fileName)
        #print(rgb_img)
        #print(vis_panoptic)    
        #cv2.imwrite(f"{outputFolder}/{fileName}",rgb_img)
        combinedFrame = np.vstack((frame, vis_panoptic))
        cv2.imwrite(f"{outputFolder}/{fileName[:-4]}.png",rgb_img)
        #cv2.imwrite(f"{outputFolder}/{fileName[:-4]}.png",combinedFrame)
        #cv2.imwrite(f"testOutput.png",combinedFrame)
        #diffTime = time.time() - startTime
        #print("segmentering:", diffTime)
        
    

inferance.writeDatasetToFile()

'''
torch.Size([1, 192, 80, 640])
torch.Size([1, 384, 40, 320])
torch.Size([1, 768, 20, 160])
torch.Size([1, 1536, 10, 80])


torch.Size([1, 192, 160, 216])
torch.Size([1, 384, 80, 108])
torch.Size([1, 768, 40, 54])
torch.Size([1, 1536, 20, 27])
'''