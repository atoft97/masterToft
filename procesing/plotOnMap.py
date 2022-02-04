import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import json
import os
import math
from tqdm import tqdm

with open("../data/LiDARnamePosDir.json") as inputFile:
    namePosDirection = json.load(inputFile)

#print(namePosDirection)

#listdir på nåken bidle

#print(imageNames)
#imageNames = [imageNames[10]]

def getPos(filename):
	return(namePosDirection[filename]['latitude'], namePosDirection[filename]['longitude'])

def getDir(filename):
	return(namePosDirection[filename]['direction']) 




image = Image.open('mapsMap.png', 'r')

BBox = ((11.4040, 11.4262, 61.1740, 61.1871))




'''
x = []
y = []
directions = []

directionX = []
directionY = []
'''

drives = os.listdir()

imageNames = os.listdir("../data/combinedImagesTaller/road_drive2")
imageNames.sort()

for imageName in tqdm(imageNames):
	fig, ax = plt.subplots(figsize = (8,7))
	#print(imageName)
	latitude, longitude = getPos(imageName)
	latitude = float(latitude)
	longitude = float(longitude)
	direction = getDir(imageName)
	direction = float(direction)
	x = longitude
	y = latitude
	directions = direction
	rad = math.radians(direction)
	x_coord = math.sin(rad)*0.002  + longitude
	y_coord = math.cos(rad)*0.002 + latitude

	#print("dir", directions)
	#print("x", x_coord)
	#print("y", y_coord)

	directionX = x_coord
	directionY = y_coord


	ax.scatter(x,y , zorder=1, c='b', s=10)

	ax.set_xlim(11.4040,11.4262)
	ax.set_ylim(61.1740,61.1871)

	ax.imshow(image, zorder=0, extent = BBox, aspect= 'equal')
	ax.annotate("", xy=(x, y), xytext=(directionX, directionY), arrowprops=dict(arrowstyle="<-")) #bytt x og y

	#plt.show()
	plt.savefig(f'plotImages3/{imageName}.png')
	plt.close()

#11.4040, 11.4262
#61.1740, 61.1871