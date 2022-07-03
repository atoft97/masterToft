import os
import json
from tqdm import tqdm

drivesPath = "/home/potetsos/lagrinatorn/master/Rellis-3D"
drives = os.listdir(drivesPath)
drives.sort()

outputPath = "/home/potetsos/lagrinatorn/master/frame2time.json"

frameNumber2Time = {}

for drive in drives:
	 filesNames = os.listdir(f"{drivesPath}/{drive}/os1_cloud_node_color_ply")
	 filesNames.sort()
	 for fileName in tqdm(filesNames):
	 	frameNumber = fileName[5:11]
	 	fullNumber = f"{drive}/{frameNumber}"


	 	time = fileName[12:-4]
	 	frameNumber2Time[fullNumber] = time

	 	
	 

with open(outputPath, 'w') as fp:
    json.dump(frameNumber2Time, fp)
