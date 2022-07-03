from inferance import Inferance
from os import listdir
from tqdm import tqdm
from detectron2.data.detection_utils import read_image
import time
import cv2

inferance = Inferance(loggingFolder="", modelName="model_0004999")

outputFolder = "outputImages"
inputfoler = "../../maskFormer/inferance/inputImages2"

files = listdir(inputfoler)

files.sort()
        
for fileName in tqdm(files):
    frame = read_image(inputfoler + "/" +fileName, format="BGR")
    #startTime = time.time()
    vis_panoptic, rgb_img, classImage = inferance.segmentImage(frame, fileName)
    #print(rgb_img)
    cv2.imwrite(f"{outputFolder}/{fileName}",rgb_img)
    #diffTime = time.time() - startTime
    #print("segmentering:", diffTime)
