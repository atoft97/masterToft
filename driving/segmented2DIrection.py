import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

inputStartPath = "drivableImages/roadDrive2/indexes"
outputStartPath = "outputDirections/roadDrive2"

innputFilenames = os.listdir(inputStartPath)
innputFilenames.sort()

print(innputFilenames)


for filename in innputFilenames:
    filePath = inputStartPath + "/" + filename
    image = Image.open(filePath)
    numpyImage = np.asarray(image)
    print(numpyImage.shape)


    break