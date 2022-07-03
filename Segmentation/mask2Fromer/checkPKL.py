from hashlib import new
import pickle
from detectron2.utils.file_io import PathManager
import numpy as np
import torch

modelPath = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/models/swinLarge.pkl"

with PathManager.open(modelPath, "rb") as f:
    data = pickle.load(f, encoding="latin1")

#print(data)

#data['']
#print(type(data))
'''
newData = {}
newData['model'] = {}

for key in data['model'].keys():
    splitted = key.split(".")
    if (splitted[0] == "backbone"):
        #print(key)
        #print(data.key)
        joined = ".".join(splitted[1:])
        newName = 'backboneRGB.' + joined
        newData['model'][newName] = data['model'][key]

for key in newData['model'].keys():
    #print(key)
    data['model'][key] = newData['model'][key]
'''

#print(data)


head1 = data['model']['sem_seg_head.pixel_decoder.input_proj.0.0.weight']
head2 = data['model']['sem_seg_head.pixel_decoder.input_proj.1.0.weight']
head3 = data['model']['sem_seg_head.pixel_decoder.input_proj.2.0.weight']
adopter = data['model']['sem_seg_head.pixel_decoder.adapter_1.weight']
#print(type(head1))
head1New = np.hstack((head1, head1))
head2New = np.hstack((head2, head2))
head3New = np.hstack((head3, head3))
adopterNew = np.hstack((adopter, adopter))

data['model']['sem_seg_head.pixel_decoder.input_proj.0.0.weight'] = head1New
data['model']['sem_seg_head.pixel_decoder.input_proj.1.0.weight'] = head2New
data['model']['sem_seg_head.pixel_decoder.input_proj.2.0.weight'] = head3New
data['model']['sem_seg_head.pixel_decoder.adapter_1.weight'] = adopterNew

#print(head1.shape)
#print(head1New.shape)





modelPath = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/models/ffrLidarFinal.pth"
#with PathManager.open(modelPath, "rb") as f:
#    data = pickle.load(f, encoding="latin1")

model = torch.load(modelPath)
#print(model)

#print(model['model'].keys())

newDataLiDAR = {}
newDataLiDAR['model'] = {}


for key in model['model'].keys():
    #print(key)
    splitted = key.split(".")
    if (splitted[0] == "backbone"):
        #print(key)
        #print(data.key)
        #joined = ".".join(splitted[1:])
        #newName = 'backbone.' + joined
        newDataLiDAR['model'][key] = model['model'][key]

for key in newDataLiDAR['model'].keys():
    #print(key)
    data['model'][key] = newDataLiDAR['model'][key]







modelPathRGB = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/models/ffiRGBLast.pth"
#with PathManager.open(modelPath, "rb") as f:
#    data = pickle.load(f, encoding="latin1")

#print(modelPathRGB['model']['backboneRGB.patch_embed.proj.weight'].shape)

modelRGB = torch.load(modelPathRGB)
#print(model)

#print(model['model'].keys())

newDataRGB = {}
newDataRGB['model'] = {}

for key in modelRGB['model'].keys():
    #print(key)
    splitted = key.split(".")
    if (splitted[0] == "backbone"):
        #print(key)
        #print(data.key)
        joined = ".".join(splitted[1:])
        newName = 'backboneRGB.' + joined
        newDataRGB['model'][newName] = modelRGB['model'][key]

for key in newDataRGB['model'].keys():
    #print(key)
    data['model'][key] = newDataRGB['model'][key]



print(data['model']['backboneRGB.patch_embed.proj.weight'].shape)
print(data['model']['backbone.patch_embed.proj.weight'].shape)

with open("models/swinLargeDubble.pkl", "wb") as f:
    pickle.dump(data , f)