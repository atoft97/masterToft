
import os
import cv2
import numpy as np
from tqdm import tqdm
#inputpath = "/home/potetsos/lagrinatorn/master/rellisOutput/combinedImagesWithLabel/labels"
inputpath = "/home/potetsos/lagrinatorn/master/rellisOutput/rgb"
#inputpath = os.listdir(inputpath)
#outputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/combinedImagesWithLabel/labelsId"

outputPath = "/home/potetsos/lagrinatorn/master/rellisOutput/rgb"


#color_to_label = {(255, 255, 255): 0, (64, 255, 38): 2, (70, 70, 70): 0, (150, 0, 191): 4, (255, 38, 38): 4, (232, 227, 81): 3, (255, 179, 0): 3, (255, 20, 20):4 , (191, 140, 0):1 , (15, 171, 255):0 , (200, 200, 200): 4, (46, 153, 0): 4, (180, 180, 180): 1}

color_to_label = {(0,0,0):0, (108,64,20):1, (0,102,0):2, (0,255,0):3,(0,153,153):4,(0,128,255):5,(0,0,255):6,(255,255,0):7,(255,0,127):8,(64,64,64):9,(255,0,0):10,(102,0,0):11,(204,153,255):12,(102,0,204):13,(255,153,204):14,(170,170,170):15,(41,121,255):16,(134,255,239):17,(99,66,34):18,(110,22,138):19}

for dataType in ["train", "test", "val"]:
	print(dataType)
	inputpathType = f"{inputpath}/{dataType}/label"
	inputpathTypePaths = os.listdir(inputpathType)
	inputpathTypePaths.sort()
	outputPathType = f"{outputPath}/{dataType}/labelsId"
	os.makedirs(outputPathType, exist_ok=True)

	for filename in tqdm(inputpathTypePaths):
		image = cv2.imread(f"{inputpathType}/{filename}")

		image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_BGR2RGB)
		#numpyImage = np.asarray(image)
		labelIdsImage = np.zeros((image.shape[0], image.shape[1]))
		for key in color_to_label.keys():
		    labelIdsImage[(image == key).all(2)] = color_to_label[key]

		cv2.imwrite(f"{outputPathType}/{filename}", labelIdsImage)