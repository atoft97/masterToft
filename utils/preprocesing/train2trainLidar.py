import os
import shutil
from tqdm import tqdm
import json

inputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/combinedImagesWithLabel"
outputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/lidarDataset"

os.makedirs(f"{outputPath}/train/image", exist_ok=True)
os.makedirs(f"{outputPath}/val/image", exist_ok=True)
os.makedirs(f"{outputPath}/test/image", exist_ok=True)

os.makedirs(f"{outputPath}/train/label", exist_ok=True)
os.makedirs(f"{outputPath}/val/label", exist_ok=True)
os.makedirs(f"{outputPath}/test/label", exist_ok=True)

os.makedirs(f"{outputPath}/train/numpyImage", exist_ok=True)
os.makedirs(f"{outputPath}/val/numpyImage", exist_ok=True)
os.makedirs(f"{outputPath}/test/numpyImage", exist_ok=True)

os.makedirs(f"{outputPath}/train/labelsId", exist_ok=True)
os.makedirs(f"{outputPath}/val/labelsId", exist_ok=True)
os.makedirs(f"{outputPath}/test/labelsId", exist_ok=True)

os.makedirs(f"{outputPath}/train/numpyCloud", exist_ok=True)
os.makedirs(f"{outputPath}/val/numpyCloud", exist_ok=True)
os.makedirs(f"{outputPath}/test/numpyCloud", exist_ok=True)

os.makedirs(f"{outputPath}/train/numpyCloud", exist_ok=True)
os.makedirs(f"{outputPath}/val/numpyCloud", exist_ok=True)
os.makedirs(f"{outputPath}/test/numpyCloud", exist_ok=True)

with open("/home/potetsos/lagrinatorn/master/frame2time.json", "r") as json_file:
    frameNumber2Time = json.load(json_file)

#print(frameNumber2Time)

for dataType in ["train", "test", "val"]:
	print("dataType", dataType)
	with open(f"/home/potetsos/lagrinatorn/master/lidarDisribution/pt_{dataType}.lst", "r") as f:
		lines = f.read().splitlines()
		#print(lines)
		#break

	for line in tqdm(lines):
		splitted = line.split(" ")
		labelName = splitted[1]
		
		labelNameList = labelName.split("/")
		#print(labelNameList)
		#labelNameList[1] = "pylon_camera_node_label_color"
		#labelName = "/".join(labelNameList)
		#filename = splitted[0]
		frameNumber = f"{labelNameList[0]}/{labelNameList[-1][0:6]}"
		#print(frameNumber)
		try:
			time = frameNumber2Time[frameNumber]
		except:
			print(time)
		#print(time)

		fullLabelPath = f"{inputPath}/labels/{time}.png"
		fullImagePath = f"{inputPath}/images/{time}.png"
		fullLabelIdPath = f"{inputPath}/labelsId/{time}.png"
		fullImageNumpyPath = f"{inputPath}/numpyImages/{time}.npy"
		fullCloudNumpyPath = f"{inputPath}/numpyCloud/{time}.npy"
		fullCloudColorNumpyPath = f"{inputPath}/numpyCloud/{time}.npy"


		#print(filename)
		#print(filename.split("/")[-1][12:])
			#newFileName = filename.split("/")[-1][12:]
			#newLabelName = labelName.split("/")[-1][12:]
		#print(fullImagePath)
		#print(f"{outputPath}/{dataType}/images/{time}.png")
		if (os.path.exists(fullImagePath)):
			#shutil.copyfile(fullImagePath, f"{outputPath}/{dataType}/image/{time}.png")
			#shutil.copyfile(fullLabelPath, f"{outputPath}/{dataType}/label/{time}.png")
			#shutil.copyfile(fullImageNumpyPath, f"{outputPath}/{dataType}/numpyImage/{time}.npy")
			#shutil.copyfile(fullCloudNumpyPath, f"{outputPath}/{dataType}/numpyCloud/{time}.npy")
			#shutil.copyfile(fullLabelIdPath, f"{outputPath}/{dataType}/labelsId/{time}.png")

			shutil.copyfile(fullCloudNumpyPath, f"{outputPath}/{dataType}/numpyCloud/{time}.npy")
			
		else:
			pass
			#print(fullImagePath)
			
	