from inferanceCombined import Inferance
from os import listdir
from tqdm import tqdm
from detectron2.data.detection_utils import read_image
import time
import cv2
import numpy as np
import os
from detectron2.data.detection_utils import read_image
from collections import OrderedDict

inferance = Inferance(loggingFolder="fraSno", modelName="cobined256150")

outputFolder = "/lhome/asbjotof/asbjotof/2022/outputImages/cobined256150"
#inputfoler = "datasetLidarTilNo/train/images"
#inputfoler = "datasetLidarTilNo/train/imagesNumpy"
#inputfoler = "inputImages/lidarDatasetNumpy"
#inputfoler = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdataset/test/numpyImage"
inputfolerLidar = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdatasetWithRGBCounter/test/lidarNumpyProjected"
inputfolerRGB = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdatasetWithRGBCounter/test/rgbProjected"
gtFolder = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdatasetWithRGBCounter/test/lidarLabelIDProjected"

os.makedirs(f"{outputFolder}/pred", exist_ok=True)
os.makedirs(f"{outputFolder}/all", exist_ok=True)
files = listdir(inputfolerLidar)

files.sort()
        
_num_classes = 14
_conf_matrix = np.zeros((_num_classes + 1, _num_classes + 1), dtype=np.int64)


for fileName in tqdm(files):
    #frame = read_image(inputfoler + "/" +fileName, format="BGR")
    #try:
    #print(fileName)
    #numpyPath = '/'.join(filePaths)[:-4] + ".npy"
    numpyPath = f"{inputfolerLidar}/{fileName}"
    numpyImage = np.load(numpyPath)

    numpyImage = numpyImage.astype("float32")



    rgbPath = f"{inputfolerRGB}/{fileName[:-4]}.png"
    rgbImage = read_image(rgbPath, format="BGR")

    #numpyImage=numpyImage.float()
    #rgbImage = rgbImage.float()

    #startTime = time.time()
    vis_panoptic, rgb_img, classImage = inferance.segmentImage(rgbImage, numpyImage, fileName)
    #print(rgb_img)
    #print(rgb_img.shape)
    #print(type(classImage))
    predClassNumpy = np.array(classImage, dtype=np.int)
    #print(f"{outputFolder}/{fileName[-4]}.png")
    #cv2.imwrite(f"{outputFolder}/{fileName[:-4]}.png", vis_panoptic)
    combinedFrame = np.vstack((rgbImage, numpyImage[:,:,0:3], vis_panoptic))
    
    cv2.imwrite(f"{outputFolder}/all/{fileName[:-4]}.png",combinedFrame)

    #diffTime = time.time() - startTime
    #print("segmentering:", diffTime)
    #except:
    #    print(fileName)
    #    numpyImage = np.load(numpyPath)
    #    print(numpyImage.shape)
    gtPath = f"{gtFolder}/{fileName[:-4]}.png"
    gtImage = cv2.imread(gtPath, -1)

    print(gtImage.shape)
    print(classImage.shape)

    

    pred = classImage
    gt = gtImage

    pred[gt==0] = 0

    _conf_matrix += np.bincount(
        (_num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
        minlength=_conf_matrix.size,
    ).reshape(_conf_matrix.shape)

    rgb_img[(gt==0)] = (0,0,0)
    cv2.imwrite(f"{outputFolder}/pred/{fileName[:-4]}.png",rgb_img)

#inferance.writeDatasetToFile()




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

_class_names = ["other"]
for element in COCO_CATEGORIES:
    _class_names.append(element['name'])



def evaluate():
    """
    Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
    * Mean intersection-over-union averaged across classes (mIoU)
    * Frequency Weighted IoU (fwIoU)
    * Mean pixel accuracy averaged across classes (mACC)
    * Pixel Accuracy (pACC)
    """
    acc = np.full(_num_classes, np.nan, dtype=np.float)
    iou = np.full(_num_classes, np.nan, dtype=np.float)
    tp = _conf_matrix.diagonal()[:-1].astype(np.float)
    pos_gt = np.sum(_conf_matrix[:-1, :-1], axis=0).astype(np.float)
    class_weights = pos_gt / np.sum(pos_gt)
    pos_pred = np.sum(_conf_matrix[:-1, :-1], axis=1).astype(np.float)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    iou_valid = (pos_gt + pos_pred) > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]

    #print(iou_valid)
    #print(np.sum(iou_valid))
    #print(iou)
    print(acc_valid)
    print(iou_valid)

    #for i in range(14):
    #    acc_valid[i] = False
    #    iou_valid[i] = False
    acc_valid[0] = False
    iou_valid[0] = False

    acc_valid[8] = False
    iou_valid[8] = False
    #iouComplete = iou[acc_valid][1:]
    #print(iouComplete)
    print(acc_valid)
    print(iou_valid)

    print(iou[acc_valid])
    print(np.sum(iou_valid))

    macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
    fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
    pacc = np.sum(tp) / np.sum(pos_gt)

    res = {}
    res["mIoU"] = 100 * miou
    res["fwIoU"] = 100 * fiou
    for i, name in enumerate(_class_names):
        res["IoU-{}".format(name)] = 100 * iou[i]
    res["mACC"] = 100 * macc
    res["pACC"] = 100 * pacc
    for i, name in enumerate(_class_names):
        res["ACC-{}".format(name)] = 100 * acc[i]

    results = OrderedDict({"sem_seg": res})
    #_logger.info(results)
    return results

res = evaluate()
print(res)