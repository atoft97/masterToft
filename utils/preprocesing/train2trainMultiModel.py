import os
import shutil
from tqdm import tqdm

inputPath = f"/home/potetsos/lagrinatorn/master/multiModal/all"
outputPath = "/home/potetsos/lagrinatorn/master/multiModal"


dataTypeDict ={"Train": "train", "Test": "test", "Validation": "val"}


for dataType in ["Train", "Test", "Validation"]:
	print("dataType", dataType)

	os.makedirs(f"{outputPath}/{dataType}/rgbImage", exist_ok=True)
	os.makedirs(f"{outputPath}/{dataType}/rgbLabelColor", exist_ok=True)
	os.makedirs(f"{outputPath}/{dataType}/rgbLabelID", exist_ok=True)

	os.makedirs(f"{outputPath}/{dataType}/lidarImage", exist_ok=True)
	os.makedirs(f"{outputPath}/{dataType}/lidarLabelColor", exist_ok=True)
	os.makedirs(f"{outputPath}/{dataType}/lidarLabelID", exist_ok=True)
	os.makedirs(f"{outputPath}/{dataType}/lidarNumpyImage", exist_ok=True)

	os.makedirs(f"{outputPath}/{dataType}/lidarPointCloud", exist_ok=True)


	#with open(f"/home/potetsos/lagrinatorn/master/cvatLablet/ffiRGBdataset/ImageSets/Segmentation/{dataType}.txt", "r") as f:
	with open(f"/home/potetsos/lagrinatorn/master/multiModal/all/{dataType}.txt", "r") as f:
		lines = f.read().splitlines()
		#print(lines)
		#break

	for filename in tqdm(lines):

		rgbImageFullPath =  f"{inputPath}/rgbImage/{filename}.png"
		rgbLabelIDFullPath =  f"{inputPath}/rgbLabelID/{filename}.png"
		rgbLabelColorFullPath =  f"{inputPath}/rgbLabelColor/{filename}.png"

		lidarImagePath = f"{inputPath}/lidarImage/{filename}.png"
		lidarLabelPath = f"{inputPath}/lidarLabelColor/{filename}.png"
		lidarLabeIDlPath = f"{inputPath}/lidarLabelID/{filename}.png"
		lidarNumpyImagePath = f"{inputPath}/lidarNumpyImage/{filename}.npy"

		lidarPoitnCloudPath = f"{inputPath}/lidarPointCloud/{filename}.npy"




		shutil.copyfile(rgbImageFullPath, f"{outputPath}/{dataType}/rgbImage/{filename}.png")
		shutil.copyfile(rgbLabelIDFullPath, f"{outputPath}/{dataType}/rgbLabelID/{filename}.png")
		shutil.copyfile(rgbLabelColorFullPath, f"{outputPath}/{dataType}/rgbLabelColor/{filename}.png")

		shutil.copyfile(lidarImagePath, f"{outputPath}/{dataType}/lidarImage/{filename}.png")
		shutil.copyfile(lidarLabelPath, f"{outputPath}/{dataType}/lidarLabelColor/{filename}.png")
		shutil.copyfile(lidarLabeIDlPath, f"{outputPath}/{dataType}/lidarLabelID/{filename}.png")
		shutil.copyfile(lidarNumpyImagePath, f"{outputPath}/{dataType}/lidarNumpyImage/{filename}.npy")

		shutil.copyfile(lidarPoitnCloudPath, f"{outputPath}/{dataType}/lidarPointCloud/{filename}.npy")





		
	