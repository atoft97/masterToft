import os
import shutil
from tqdm import tqdm

datasetName = "ffiLiDARdataset"

inputPath = f"/home/potetsos/lagrinatorn/master/cvatLablet/{datasetName}"
outputPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdataset"

os.makedirs(f"{outputPath}/train/image", exist_ok=True)
os.makedirs(f"{outputPath}/val/image", exist_ok=True)
os.makedirs(f"{outputPath}/test/image", exist_ok=True)

os.makedirs(f"{outputPath}/train/label", exist_ok=True)
os.makedirs(f"{outputPath}/val/label", exist_ok=True)
os.makedirs(f"{outputPath}/test/label", exist_ok=True)

dataTypeDict ={"Train": "train", "Test": "test", "Validation": "val"}

for dataType in ["Train", "Test", "Validation"]:
	print("dataType", dataType)

	if (os.path.exists(f"/home/potetsos/lagrinatorn/master/cvatLablet/{datasetName}/ImageSets/Segmentation/{dataType}.txt")):

		#with open(f"/home/potetsos/lagrinatorn/master/cvatLablet/ffiRGBdataset/ImageSets/Segmentation/{dataType}.txt", "r") as f:
		with open(f"/home/potetsos/lagrinatorn/master/cvatLablet/{datasetName}/ImageSets/Segmentation/{dataType}.txt", "r") as f:
			lines = f.read().splitlines()
			#print(lines)
			#break

		for filename in tqdm(lines):
			fullImagePath = f"{inputPath}/JPEGImages/{filename}.png"
			fullLabelPath = f"{inputPath}/SegmentationClass/{filename}.png"

			shutil.copyfile(fullImagePath, f"{outputPath}/{dataTypeDict[dataType]}/image/{filename}.png")
			shutil.copyfile(fullLabelPath, f"{outputPath}/{dataTypeDict[dataType]}/label/{filename}.png")
	else:
		print("not found:", f"/home/potetsos/lagrinatorn/master/cvatLablet/{datasetName}ImageSets/Segmentation/{dataType}.txt")

		
	