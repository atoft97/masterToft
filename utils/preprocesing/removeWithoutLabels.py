import os
import shutil
from tqdm import tqdm


drivesPath = "/home/potetsos/lagrinatorn/master/rellisOutput/combinedImages"
drives = os.listdir(drivesPath)
drives.sort()

drivesNumpyPath = "/home/potetsos/lagrinatorn/master/rellisOutput/combinedNumpy"
drivesNumpy = os.listdir(drivesPath)
drivesNumpy.sort()

outputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/combinedImagesWithLabel"
labelsPath ="/home/potetsos/lagrinatorn/master/rellisOutput/labels"
pointCloudPath ="/home/potetsos/lagrinatorn/master/rellisOutput/numpyPointcloud"
os.makedirs(f"{outputPath}/labels", exist_ok=True)
os.makedirs(f"{outputPath}/images", exist_ok=True)
os.makedirs(f"{outputPath}/numpyImages", exist_ok=True)
os.makedirs(f"{outputPath}/numpyCloud", exist_ok=True)

drivesLabel = os.listdir(labelsPath)
drivesLabel.sort()

'''
for drive in drives:
	driveFullPath = f"{drivesPath}/{drive}"
	combinedPaths = os.listdir(driveFullPath)
	combinedPaths.sort()
	labelsDrivePaths = os.listdir(f"{labelsPath}/{drive}")
	for combinedPath in tqdm(combinedPaths):
		fullCombinedPath = f"{driveFullPath}/{combinedPath}"
		#print(combinedPath)
		if (combinedPath in labelsDrivePaths):
			#kopier til 
			#f"{outputPath}/{drive}/{combinedPath}"
			shutil.copyfile(fullCombinedPath, f"{outputPath}/images/{combinedPath}")

'''
for drive in drives:
	driveFullPath = f"{labelsPath}/{drive}"
	pointPath = f"{pointCloudPath}/{drive}"
	labelPaths = os.listdir(driveFullPath)
	labelPaths.sort()
	if (os.path.exists(f"{drivesPath}/{drive}")):
		combinedPathPaths = os.listdir(f"{drivesPath}/{drive}")
		for labelPath in tqdm(labelPaths):
			fullLabelPath = f"{driveFullPath}/{labelPath}"
			#print(combinedPath)
			pointFullPath = f"{pointPath}/{labelPath[:-4]}.npy"
			if (labelPath in combinedPathPaths):
				#kopier til 
				#f"{outputPath}/{drive}/{combinedPath}"
				shutil.copyfile(fullLabelPath, f"{outputPath}/labels/{labelPath}")
				shutil.copyfile(pointFullPath, f"{outputPath}/numpyCloud/{labelPath[:-4]}.npy")
'''
for driveNumpy in drivesNumpy:
	driveFullPath = f"{drivesNumpyPath}/{driveNumpy}"
	combinedPaths = os.listdir(driveFullPath)
	combinedPaths.sort()
	labelsDrivePaths = os.listdir(f"{labelsPath}/{driveNumpy}")
	for combinedPath in tqdm(combinedPaths):
		fullCombinedPath = f"{driveFullPath}/{combinedPath}"
		if (f"{combinedPath[:-4]}.png" in labelsDrivePaths):
			#kopier til 
			#f"{outputPath}/{drive}/{combinedPath}"
			shutil.copyfile(fullCombinedPath, f"{outputPath}/numpyImages/{combinedPath}")
'''