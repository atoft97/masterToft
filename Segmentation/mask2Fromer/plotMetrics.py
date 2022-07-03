from dataclasses import field
import json
import matplotlib.pyplot as plt
from csv import DictWriter

modelName = "ffiLidarRange16"

inputPath = f"/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/{modelName}/metrics.json"

trainLossX = []
trainLossY = []
valmIoUX = []
valmIoUY = []

valLossX = []
valLossY = []

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
{"supercategory": "Snow", 		"color": [203,203,255],  "isthing": 0, "id": 13, "name": "Snow"}
]

#categories = ["sky", "grass", "tree", "bush", "concrete", "mud", "person", "puddle", "rubble", "barrier", "log", "fence", "vehicle", "object", "pole", "water", "asphalt", "building"]
categories = ["Forrest", "CameraEdge", "Vehicle", "Person", "Bush", "Puddle", "Dirtroad", "Sky", "Large_stone", "Grass", "Gravel", "Building", "Snow"]
#categories = ['void', 'dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object', 'asphalt', 'building', 'log', 'person', 'fence', 'bush', 'concrete', 'barrier', 'puddle', 'mud', 'rubble']
#categories = ["Forrest", "CameraEdge", "Vehicle", "Person", "Bush", "Puddle", "Dirtroad", "Sky", "Large_stone", "Grass", "Gravel", "Building"]
fieldNamesAcc = []
for element in categories:
    fieldNamesAcc.append(f"sem_seg/IoU-{element}")

def write_to_csv(new_data, fileName):
    with open(fileName, 'a') as f_object:
        writer_object = DictWriter(f_object, fieldnames=categories)
        #print(new_data)
        writer_object.writerow(new_data)
        print("skriv til fil:", fileName)
        f_object.close()

headers = {}
for element in categories:
    headers[element] = element
write_to_csv(headers, f"{modelName}/test.csv")

def addTrainLossLine(lossLine):
    trainLossX.append(lossLine['iteration'])
    trainLossY.append(lossLine['total_loss'])
    
    try:
        valLossY.append(lossLine['validation_loss flott'])
        valLossX.append(lossLine['iteration'])
    except:
        print("ingen val loss")
    
    

def addValInfo(lossLine):
    print("adding val")
    valmIoUX.append(lossLine['iteration'])
    valmIoUY.append(lossLine['sem_seg/mIoU'])

    

    #valLossX.append()

    accJson = {}
    for i, element in enumerate(categories):
        accJson[element] = lossLine[fieldNamesAcc[i]]
    
    write_to_csv(accJson, f"{modelName}/test.csv")


    

    

with open(inputPath, "r") as f:
    for line in f.readlines():
        #print(line)
        lineJson = json.loads(line)
        if ('total_loss' in lineJson.keys()):
            addTrainLossLine(lineJson)
        if ('sem_seg/mIoU' in lineJson.keys()):
            addValInfo(lineJson)


plt.plot(valmIoUX, valmIoUY, label="Validation IoU")
plt.plot(trainLossX, trainLossY, label="Train loss")
plt.plot(valLossX, valLossY, label="Validation loss")

plt.title("Training and validation loss")
plt.xlabel("Iteration")
plt.ylabel("Loss/IoU")
plt.legend()
plt.savefig(f"{modelName}/trainValLoss.png")
plt.show()


