import os
from tqdm import tqdm
import shutil

inputpath = "/home/potetsos/lagrinatorn/master/ffiLiDARdataset16"
outputPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdataset16"
inputNumpyPath = "/home/potetsos/skule/2022/masterCode/masterToft/data/combinedNumpy16/all"

for dataType in ["train", "test", "val"]:
	print(dataType)
	inputpathType = f"{inputpath}/{dataType}/idSegment"
	inputpathTypePaths = os.listdir(inputpathType)
	inputpathTypePaths.sort()
	outputPathNumpy = f"{outputPath}/{dataType}/numpyImage"
	os.makedirs(f"{outputPath}/{dataType}/numpyImage", exist_ok=True)

	for filename in tqdm(inputpathTypePaths):
		#image = cv2.imread(f"{inputpathType}/{filename}")
		fullNumpyPath = f"{inputNumpyPath}/{filename[:-4]}.npy"
		shutil.copyfile(fullNumpyPath, f"{outputPath}/{dataType}/numpyImage/{filename[:-4]}.npy")