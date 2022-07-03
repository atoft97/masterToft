import numpy as np
import cv2

import os

for dataType in ["train", "val", "test"]:
    starPath = f"/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdataset/{dataType}/numpyImage"
    outputPath = f"/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdataset/{dataType}/3channelImage"

    os.makedirs(outputPath, exist_ok=True)

    for filename in os.listdir(starPath):
        imagePath = f"{starPath}/{filename}"

        numpyImage = np.load(imagePath)

        a = numpyImage[:,:,0]
        b = numpyImage[:,:,2:4]
        c=np.dstack((a,b))
        #c = a+b
        print(c.shape)

        cv2.imwrite(f"{outputPath}/{filename[:-4]}.png", numpyImage[:,:,0:3])
        #print(f"{outputPath}/{filename[:-4]}.png")