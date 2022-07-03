import os
import shutil
from tqdm import tqdm

inputPath = "/home/potetsos/lagrinatorn/master/Rellis-3D"
outputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/rgb"

os.makedirs(f"{outputPath}/train/image", exist_ok=True)
os.makedirs(f"{outputPath}/val/image", exist_ok=True)
os.makedirs(f"{outputPath}/test/image", exist_ok=True)

os.makedirs(f"{outputPath}/train/label", exist_ok=True)
os.makedirs(f"{outputPath}/val/label", exist_ok=True)
os.makedirs(f"{outputPath}/test/label", exist_ok=True)

for dataType in ["train", "test", "val"]:
	print("dataType", dataType)
	with open(f"/home/potetsos/lagrinatorn/master/{dataType}.lst", "r") as f:
		lines = f.read().splitlines()
		#print(lines)
		#break

	for line in tqdm(lines):
		splitted = line.split(" ")
		labelName = splitted[1]
		#print(labelName)
		labelNameList = labelName.split("/")
		labelNameList[1] = "pylon_camera_node_label_color"
		labelName = "/".join(labelNameList)
		filename = splitted[0]

		fullLabelPath = f"{inputPath}/{labelName}"
		fullImagePath = f"{inputPath}/{filename}"

		#print(filename)
		#print(filename.split("/")[-1][12:])
		newFileName = filename.split("/")[-1][12:]
		newLabelName = labelName.split("/")[-1][12:]

		shutil.copyfile(fullImagePath, f"{outputPath}/{dataType}/image/{newFileName}")
		shutil.copyfile(fullLabelPath, f"{outputPath}/{dataType}/label/{newLabelName}")

		
	