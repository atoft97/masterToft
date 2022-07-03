import os
import numpy as np
import cv2
from tqdm import tqdm

lidarDatasetPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdataset"
outputDatasetPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetSmaller"

for dataType in ["train", "test", "val"]:
    print(dataType)
    idSegmentPath = f"{lidarDatasetPath}/{dataType}/idSegment"

    os.makedirs(f"{outputDatasetPath}/{dataType}/idSegment/", exist_ok=True)
    os.makedirs(f"{outputDatasetPath}/{dataType}/image/", exist_ok=True)
    os.makedirs(f"{outputDatasetPath}/{dataType}/label/", exist_ok=True)
    os.makedirs(f"{outputDatasetPath}/{dataType}/numpyImage/", exist_ok=True)

    for imageName in tqdm(os.listdir(idSegmentPath)):

        idSegment = cv2.imread(f"{lidarDatasetPath}/{dataType}/idSegment/{imageName}", -1)
        image = cv2.imread(f"{lidarDatasetPath}/{dataType}/image/{imageName}", -1)
        label = cv2.imread(f"{lidarDatasetPath}/{dataType}/label/{imageName}", -1)
        numpyImage = np.load(f"{lidarDatasetPath}/{dataType}/numpyImage/{imageName[:-4]}.npy")

        dim = (2048, 64)
        idSegment = cv2.resize(idSegment, dim, interpolation = cv2.INTER_NEAREST)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
        label = cv2.resize(label, dim, interpolation = cv2.INTER_NEAREST)
        numpyImage = cv2.resize(numpyImage, dim, interpolation = cv2.INTER_NEAREST)

        cv2.imwrite(f"{outputDatasetPath}/{dataType}/idSegment/{imageName}", idSegment)
        cv2.imwrite(f"{outputDatasetPath}/{dataType}/image/{imageName}", image)
        cv2.imwrite(f"{outputDatasetPath}/{dataType}/label/{imageName}", label)
        np.save(f"{outputDatasetPath}/{dataType}/numpyImage/{imageName[:-4]}.npy", numpyImage)