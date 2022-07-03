import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from os import listdir

from torchvision.transforms import Compose

from detectron2.modeling import build_model

import json
from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_panoptic
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

import nvidia_smi

from mask2former import add_maskformer2_config
dirname = os.path.dirname(__file__)

import cProfile
import pstats
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

from registerDataset import register_ffi_png_dataset

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

'''
model = build_model(cfg)
model.eval()

checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)


aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

input_format = cfg.INPUT.FORMAT



def predictImage(frame):

    with torch.no_grad():
        height, width = frame.shape[:2]
        image = aug.get_transform(frame).apply_image(frame)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
    	
    #	print(inputs)
        predictions = model([inputs])
        predictions = predictions[0]

    predictedPanoptic = predictions

    return(predictedPanoptic)
'''

class Inferance:

    def __init__(self, loggingFolder, modelName):
        self.loggingFolder = loggingFolder
        if (loggingFolder != ""):
            self.printGPUInfo()
        self.modelName = modelName
        #self.modelName = "semanticRGB"
        self.device = torch.device("cuda")
        

        self.registerDataset()
        self.cfg = self.createConfig()
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

        self.video_visualizer = VideoVisualizer(self.metadata, ColorMode.IMAGE)
        self.predictor = DefaultPredictor(self.cfg)

        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)


        self.aug = T.ResizeShortestEdge([self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST)

        self.input_format = self.cfg.INPUT.FORMAT

        self.index = 0
        
        #imageStartPath = os.path.join(dirname, "inputImages2")
        #imageStartPath = imageStartPath
        #self.startPath = imageStartPath
        #self.files = listdir(self.startPath)
        #self.files.sort()

        self.counter=0
        self.totalTime = 0
        self.totalTimeCounter = 0

        self.lable_to_color = self.createLabelColorDict()

        if (loggingFolder != ""):
            self.coco = self.initOutputCocoDataset()
        

    def printGPUInfo(self):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Total memory:", info.total)
        print("Free memory:", info.free)
        print("Used memory:", info.used)

    def registerDataset(self):
        '''
        register_coco_panoptic_separated(name="ffi_train", metadata={}, sem_seg_root="../../../data/dataset/train/panoptic_stuff_train", image_root="../../../data/dataset/train/images", panoptic_root="../../../data/dataset/train/panoptic_train" , panoptic_json="../../../data/dataset/train/annotations/panoptic_train.json" ,instances_json="../../../data/dataset/train/annotations/instances_train.json")
        register_coco_panoptic_separated(name="ffi_val", metadata={}, sem_seg_root="../../../data/dataset/train/panoptic_stuff_train", image_root="../../../data/dataset/train/images", panoptic_root="../../../data/dataset/train/panoptic_train" , panoptic_json="../../../data/dataset/train/annotations/panoptic_train.json" ,instances_json="../../../data/dataset/train/annotations/instances_train.json")

        MetadataCatalog.get("ffi_train_stuffonly").set(stuff_classes=['other', 'Grass', 'CameraEdge', 'Vehicle', 'Person', 'Bush', 'Puddle', 'Building', 'Dirtroad', 'Sky', 'Large_stone', 'Forrest', 'Gravel'])
        MetadataCatalog.get("ffi_train_stuffonly").set(stuff_colors=[[255,255,255], [64,255,38], [70,70,70], [150,0,191], [255,38,38], [232,227,81], [255,179,0], [255,20,20], [191,140,0], [15,171,255], [200,200,200], [46,153,0]])

        MetadataCatalog.get("ffi_val_stuffonly").set(stuff_classes=['other', 'Grass', 'CameraEdge', 'Vehicle', 'Person', 'Bush', 'Puddle', 'Building', 'Dirtroad', 'Sky', 'Large_stone', 'Forrest', 'Gravel'])
        MetadataCatalog.get("ffi_val_stuffonly").set(stuff_colors=[[255,255,255], [64,255,38], [70,70,70], [150,0,191], [255,38,38], [232,227,81], [255,179,0], [255,20,20], [191,140,0], [15,171,255], [200,200,200], [46,153,0], [180,180,180]])
        '''
        register_ffi_png_dataset(name="ffi_train_stuffonly", metadata={}, sem_seg_root="ffiRGBdataset/train/idSegment", image_root="ffiRGBdataset/train/image")
        register_ffi_png_dataset(name="ffi_val_stuffonly", metadata={}, sem_seg_root="ffiRGBdataset/val/idSegment", image_root="ffiRGBdataset/val/image")

        classes = ["other"]
        colors = [[255,255,255]]

        for category in COCO_CATEGORIES:
            classes.append(category['name'])
            colors.append(category['color'])

        MetadataCatalog.get("ffi_train_stuffonly").set(stuff_classes=classes)
        MetadataCatalog.get("ffi_train_stuffonly").set(stuff_colors=colors)

        MetadataCatalog.get("ffi_val_stuffonly").set(stuff_classes=classes)
        MetadataCatalog.get("ffi_val_stuffonly").set(stuff_colors=colors)

    def createConfig(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        swinConfigPath = os.path.join(dirname, "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml")
        #cfg.merge_from_file("configs/ade20k/semantic-segmentation/swin/maskformer2_swin_small_bs16_160k.yaml")
        cfg.merge_from_file(swinConfigPath)

        modelPath = os.path.join(dirname, "models/" + self.modelName +".pth")
        cfg.MODEL.WEIGHTS = modelPath

        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 14
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

        cfg.DATASETS.TRAIN = ("ffi_train_stuffonly", )
        cfg.DATASETS.TEST = ("ffi_val_stuffonly", )
        cfg.MODEL.PIXEL_MEAN = [128,128,128,128]
        cfg.MODEL.PIXEL_STD = [58,58,58,58]
        #obs obs
        #cfg.MODEL.PIXEL_MEAN = [128,128,128,128,128,128,128]
        #cfg.MODEL.PIXEL_STD = [58,58,58,58, 58,58,58]

        #cfg.freeze()
        print(cfg.INPUT.MIN_SIZE_TEST)
        print(cfg.INPUT.MAX_SIZE_TEST)
        return(cfg)

    def visualise_predicted_frame(self, frame, predictions, fileName):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_visualizer = Visualizer(frame, self.metadata)
        sem_seg = predictions["sem_seg"]

        classesImage = sem_seg.argmax(dim=0).to('cpu')
        #classesImage = cv2.imread("ffiLidarAlmostDataset/train/idSegment/stand_still_short00047.png", -1)
        #print(f"ffiLiDARdataset/test/idSegment/{fileName[-4]}")
        #classesImage = cv2.imread(f"ffiLiDARdataset/test/idSegment/{fileName[:-4]}.png", -1)
        #print("bra", classesImage)
        frame_visualizer.draw_sem_seg(classesImage, area_threshold=None, alpha=0.5)
        vis_frame = frame_visualizer.output
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        return(vis_frame)

    def createLabelColorDict(self):
        lable_to_color = {}
        colorCounter = 0
        for color in self.metadata.get('stuff_colors'):
            lable_to_color[colorCounter] = color
            colorCounter += 1
        return(lable_to_color)


    def visEachClass(self, frame, predictions, fileName):
        #print("lidarBildeShape", frame.shape)
        if (self.loggingFolder != ""):
            os.makedirs("outputImages/LiDAR/" + self.loggingFolder, exist_ok=True)
            os.makedirs("outputImages/LiDAR/" + self.loggingFolder + "/" + "outputDatasett" + "/SegmentationClass", exist_ok=True)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sem_seg = predictions["sem_seg"]
        classImage = sem_seg.argmax(dim=0).to('cpu')

        if (self.loggingFolder != ""):
            os.makedirs("outputImages/LiDAR/" + self.loggingFolder + "/" + fileName[:-4], exist_ok=True)
            for classNumber in range(len(sem_seg)):
                pred = (sem_seg[classNumber] * 255).to('cpu').numpy()
                cv2.imwrite("outputImages/LiDAR/" + self.loggingFolder + "/" + fileName[:-4] + "/" + str(classNumber) + ".png", pred)

        rgb_img = np.zeros((*classImage.shape, 3))
        for key in self.lable_to_color.keys():
            rgb_img[classImage == key] = self.lable_to_color[key]
        rgb_img = cv2.cvtColor(rgb_img.astype('float32'), cv2.COLOR_BGR2RGB)

        if (self.loggingFolder != ""):
            cv2.imwrite("outputImages/LiDAR/" + self.loggingFolder+ "/" + "outputDatasett" + "/SegmentationClass/" + fileName[:-3] + "png", rgb_img)
        return(rgb_img, classImage)



    def writeLabelMap(self):
        os.makedirs("outputImages/LiDAR/" + self.loggingFolder + "/" + "outputDatasett", exist_ok=True)
        with open("outputImages/LiDAR/" + self.loggingFolder + "/" + "outputDatasett" + "/labelmap.txt", 'w') as f:
            for i in range(len(self.metadata.get("stuff_classes"))):
                #print("flott")
                #print(metadata.get("stuff_classes")[i])
                f.write(str(self.metadata.get("stuff_classes")[i]))
                f.write(":")
                #print(str(metadata.get("stuff_colors")[i]))
                f.write(str(self.metadata.get("stuff_colors")[i][0]))
                f.write(",")
                f.write(str(self.metadata.get("stuff_colors")[i][1]))
                f.write(",")
                f.write(str(self.metadata.get("stuff_colors")[i][2]))
                f.write(":")
                f.write(":")
                f.write('\n')
    


    def detectronToCoco(self, predictions, iamgeID, startID):
        sem_seg = predictions["sem_seg"]
        classImage = sem_seg.argmax(dim=0).to('cpu')
        tensors = []
        for classNumber in range(1, len(sem_seg) +1):
            class_tensor = torch.where(classImage == classNumber, 1, 0)
            tensors.append(class_tensor)

        stacked = torch.stack(tensors)
        stackedNumpy = stacked.cpu().detach().numpy()

        annotations = []
        #negtiveSegment = 0
        for segment_number in range((len(sem_seg))):
            segmentDict = {'id': segment_number+startID}

            segmentDict['category_id'] = segment_number +1
            
            segmentDict['image_id'] = iamgeID
            segmentDict['area'] = 0
            segmentDict['bbox'] = [0,0,0,0]
            segmentDict['iscrowd'] = 0

            border = cv2.copyMakeBorder((stackedNumpy[segment_number]).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            contours, hierarchy = cv2.findContours(border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation_list = []

            for countor in contours:
                x_y_list = []
                for coordinates in countor:
                    x_y_list.append(int(coordinates[0][0]))
                    x_y_list.append(int(coordinates[0][1]))
                if (len(x_y_list)>4):
                    segmentation_list.append(x_y_list)
            #if (len(segmentation_list) > 0):
            segmentDict['segmentation'] = segmentation_list
            annotations.append(segmentDict)
            #negtiveSegment += 1
        return(annotations)

    def metadatToCoco(self, metadata):
        categories = []
        for i in range(1, len(metadata.stuff_classes)):
            category = {'id': i, 'name': metadata.stuff_classes[i], "supercategory": ""}
            categories.append(category)
        return(categories)

    def initOutputCocoDataset(self):
        self.images = []
        self.segmentID = 1
        
        self.annotations = []

        self.writeLabelMap()

        coco = {"licenses": [
            {
            "name": "",
            "id": 0,
            "url": ""
            }],  
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": ""
        },
        }
        coco['categories'] = self.metadatToCoco(self.metadata)
        return(coco)

    def writeDatasetToFile(self):
        self.coco['images'] = self.images
        self.coco['annotations'] = self.annotations
        with open("outputImages/LiDAR/" + self.loggingFolder + "/" + "outputDatasett/anotationsFFI.json", 'w') as fp:
            json.dump(self.coco, fp)

    def predictImage(self,rgbImage, lidarImage):
    #def predictImage(self,image):
    #obs obs   

        with torch.no_grad():
            height, width = rgbImage.shape[:2]
            #height, width = image.shape[:2]

            #image = self.aug.get_transform(frame).apply_image(frame)
            #image = frame
            #cv2.imwrite("forNetverk.png", image)
            #print("for for nettverk", image.shape)
            rgbImage = torch.as_tensor(rgbImage.astype("float32").transpose(2, 0, 1))
            lidarImage = torch.as_tensor(lidarImage.astype("float32").transpose(2, 0, 1))

            #image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))


            #print("for nettverk", image.shape)
            inputs = {"rgbImage": rgbImage, "image":lidarImage, "height": height, "width": width}
            #inputs = {"image":image, "height": height, "width": width}
            #inputs["rgbImage"] = rgbImage
            #inputs['image'] = image
            
        #	print(inputs)
            #print(inputs['image'].shape)
            startTime = time.time()
            #print("storrelse", image.shape)
            predictions = self.model([inputs])
            diffTime = time.time() - startTime
            print("fuksjonstid:", diffTime)
            self.totalTime += diffTime
            self.totalTimeCounter += 1
            print("total avg", self.totalTime/self.totalTimeCounter)


            #predictions = self.model([inputs])
            predictions = predictions[0]

        predictedPanoptic = predictions

        return(predictedPanoptic)

    #for fileName in tqdm(files):
    def segmentImage(self, rgbImage, lidarImage, fileName):

        
        
        self.counter += 1
        #predictedPanoptic = self.predictor([image, image])
        #combinedImage = np.dstack((lidarImage, rgbImage))
        predictedPanoptic = self.predictImage(rgbImage, lidarImage)
        #predictedPanoptic = self.predictImage(combinedImage)
        #obs obs split
        #print(predictedPanoptic.shape)

        #startTime = time.time()
        rgb_img, classImage = self.visEachClass(lidarImage, predictedPanoptic, fileName)
        #diffTime = time.time() - startTime
        #print("fuksjonstid hei:", diffTime)
        
        if (self.loggingFolder != ""):
            #print(self.loggingFolder)
            os.makedirs("outputImages/LiDAR/" + self.loggingFolder + "/" + "outputDatasett/ImageSets/Segmentation", exist_ok=True) #move to init
            with open("outputImages/LiDAR/" + self.loggingFolder + "/" + "outputDatasett/ImageSets/Segmentation" + "/default.txt", 'a') as f:
                f.write(fileName[:-4])
                f.write('\n')

            vis_panoptic = self.visualise_predicted_frame(rgbImage, predictedPanoptic, fileName)
            #print(image[:,:,0:3].shape)
            #print(vis_panoptic.shape)
            combinedFrame = np.vstack((vis_panoptic, lidarImage[:,:,0:3]))
            cv2.imwrite("outputImages/LiDAR/" + self.loggingFolder + "/" + fileName[:-4] + ".png", combinedFrame)

            height, width = lidarImage.shape[:2]
            imageCoco = {"id": self.counter, "width": width, "height": height, "file_name": fileName[:-3] + "png", "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}
            self.images.append(imageCoco)
            annotationsImage = self.detectronToCoco(predictedPanoptic, self.counter, self.segmentID)
            self.annotations.extend(annotationsImage)
            self.segmentID += len(annotationsImage)
        
        else:
            vis_panoptic = None



        return(vis_panoptic, rgb_img, classImage)

