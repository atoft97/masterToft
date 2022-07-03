
import os
import cv2
import numpy as np
from tqdm import tqdm
#inputpath = "/home/potetsos/lagrinatorn/master/rellisOutput/combinedImagesWithLabel/labels"
inputpath = "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset"
#inputpath = os.listdir(inputpath)
#outputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/combinedImagesWithLabel/labelsId"

outputPath = "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset"


#color_to_label = {(255, 255, 255): 0, (64, 255, 38): 2, (70, 70, 70): 0, (150, 0, 191): 4, (255, 38, 38): 4, (232, 227, 81): 3, (255, 179, 0): 3, (255, 20, 20):4 , (191, 140, 0):1 , (15, 171, 255):0 , (200, 200, 200): 4, (46, 153, 0): 4, (180, 180, 180): 1}

#color_to_label = {(0,0,0):0, (108,64,20):1, (0,102,0):2, (0,255,0):3,(0,153,153):4,(0,128,255):5,(0,0,255):6,(255,255,0):7,(255,0,127):8,(64,64,64):9,(255,0,0):10,(102,0,0):11,(204,153,255):12,(102,0,204):13,(255,153,204):14,(170,170,170):15,(41,121,255):16,(134,255,239):17,(99,66,34):18,(110,22,138):19}


COCO_CATEGORIES = [
{"supercategory": "Forrest",    "color": [46,153,0],    "isthing": 0, "id": 1,  "name": "Forrest"},
{"supercategory": "CameraEdge", "color": [70,70,70],    "isthing": 0, "id": 2,  "name": "CameraEdge"},
{"supercategory": "Vehicle",    "color": [150,0,191],   "isthing": 0, "id": 3,  "name": "Vehicle"},
{"supercategory": "Person",     "color": [255,38,38],   "isthing": 0, "id": 4,  "name": "Person"},
{"supercategory": "Bush",       "color": [232,227,81],  "isthing": 0, "id": 5,  "name": "Bush"},
{"supercategory": "Puddle",     "color": [255,179,0],   "isthing": 0, "id": 6,  "name": "Puddle"},
{"supercategory": "Dirtroad",   "color": [191,140,0],   "isthing": 0, "id": 7,  "name": "Dirtroad"},
{"supercategory": "Sky",        "color": [15,171,255],  "isthing": 0, "id": 8,  "name": "Sky"},
{"supercategory": "Large_stone","color": [200,200,200], "isthing": 0, "id": 9,  "name": "Large_stone"},
{"supercategory": "Grass",      "color": [64,255,38],   "isthing": 0, "id": 10, "name": "Grass"},
{"supercategory": "Gravel",     "color": [180,180,180], "isthing": 0, "id": 11, "name": "Gravel"},
{"supercategory": "Building",   "color": [255,20,20],   "isthing": 0, "id": 12, "name": "Building"},
{"supercategory": "background",   "color": [0,0,0],   "isthing": 0, "id": 13, "name": "background"}
]

color_to_label = {}
for element in COCO_CATEGORIES:
	color_to_label[tuple(element["color"])] = element["id"]

print(color_to_label)

for dataType in ["train", "test", "val"]:
	for sensor in ["lidar", "rgb"]:
		print(dataType)
		inputpathType = f"{inputpath}/{dataType}/colorSegment/{sensor}"
		inputpathTypePaths = os.listdir(inputpathType)
		inputpathTypePaths.sort()
		outputPathType = f"{outputPath}/{dataType}/idSegment/{sensor}"
		os.makedirs(outputPathType, exist_ok=True)

		for filename in tqdm(inputpathTypePaths):
			image = cv2.imread(f"{inputpathType}/{filename}")

			image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_BGR2RGB)
			#numpyImage = np.asarray(image)
			labelIdsImage = np.zeros((image.shape[0], image.shape[1]))
			for key in color_to_label.keys():
			    labelIdsImage[(image == key).all(2)] = color_to_label[key]

			cv2.imwrite(f"{outputPathType}/{filename}", labelIdsImage)
			