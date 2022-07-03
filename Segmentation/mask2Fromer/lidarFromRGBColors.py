from inferance import Inferance
#from inferanceRellisRGB import Inferance
#from inferanceRellisLidar import InferanceRellisLidar
from os import listdir, makedirs
from tqdm import tqdm
from detectron2.data.detection_utils import read_image
import time
import cv2
import numpy as np
#from projectRGBtoLidar1Channel import lidarRGBcombine
from projectRGBtoLidar import lidarRGBcombine
from mask_former_semantic_dataset_mapper_test import MaskFormerSemanticDatasetMapperTest
from registerDataset import register_numpy_dataset

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
    inference_on_dataset,
)
from collections import OrderedDict


#inferance = Inferance(loggingFolder="slettDenne", modelName="rellisBRG1t")
#inferance = Inferance(loggingFolder="test2", modelName="ffiRGButenOther")
inferance = Inferance(loggingFolder="lidarFromRGB", modelName="ffiRGBLast")

#inferance = Inferance(loggingFolder="test2", modelName="ffiRGBfinal")


#inferance = InferanceRellisLidar(loggingFolder="slettDenneLidar", modelName="rellisLidar")


#outputFolder = "outputImages/Rellis/lidar"
#outputFolders = "/lhome/asbjotof/asbjotof/2022/lidarFromRGBSegmented"
#inputfoler = "../../maskFormer/inferance/inputImages2"
#inputfoler = "rgb/test/image"
    #innputFoldersPath = "/lhome/asbjotof/asbjotof/2022/bilder/rgb"
    #inputfolers = listdir(innputFoldersPath)
    #inputfolers.sort()
#inputfoler = "lidarDataset/test/numpyImage"

#inputfolerFullPath = f"{innputFoldersPath}/{inputfoler}"
inputfolerPath = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdatasetWithRGBCounter/test/"
inputfolerFullPath = "/lhome/asbjotof/asbjotof/2022/masterToft/LiDARsegmentation/mask2Former/inferance/ffiLiDARdatasetWithRGBCounter/test/rgbImage"
#outputFolder = f"{outputFolders}/{inputfoler}"
outputFolder = "/lhome/asbjotof/asbjotof/2022/lidarFromRGBSegmented"

lidarSegmentedPath = "/lhome/asbjotof/asbjotof/2022/lidarFromRGBSegmented/lidarPred"

files = listdir(inputfolerFullPath)

files.sort()
makedirs(f"{outputFolder}/rgbSegmented", exist_ok=True)
makedirs(f"{outputFolder}/visualisert", exist_ok=True)
makedirs(f"{outputFolder}/labelFromRGB", exist_ok=True)
makedirs(f"{outputFolder}/gt", exist_ok=True)
makedirs(f"{outputFolder}/pred", exist_ok=True)
makedirs(f"{outputFolder}/lidarPredCut", exist_ok=True)


_num_classes = 14
_conf_matrix = np.zeros((_num_classes + 1, _num_classes + 1), dtype=np.int64)

for fileName in tqdm(files):
    #frame = read_image("/lhome/asbjotof/asbjotof/2022/bilder/rgb/stand_still_short/stand_still_short_00000.png", format="BGR")
    frame = read_image(inputfolerFullPath + "/" +fileName, format="BGR")

    #frame = read_image("ffiRGBdataset/train/image/2022-04-08-13-49-07_00400.png", format="BGR")

    #frame = np.load(f"{inputfoler}/{fileName}")
    #startTime = time.time()
    vis_panoptic, rgb_img, classImage = inferance.segmentImage(frame, fileName)
    #print(rgb_img)
    #print(vis_panoptic)    
    #cv2.imwrite(f"{outputFolder}/{fileName}",rgb_img)
    combinedFrame = np.hstack((frame, vis_panoptic))
    #print(fileName)
    #print(f"{outputFolder}/{fileName[:-4]}.png")
    cv2.imwrite(f"{outputFolder}/rgbSegmented/{fileName[:-4]}.png",combinedFrame)
    #cv2.imwrite(f"testOutput.png",combinedFrame)
    #diffTime = time.time() - startTime
    #print("segmentering:", diffTime)

    rgbFullPath = inputfolerFullPath + "/" +fileName
    
    lidarImageFullPath = f"{inputfolerPath}/lidarImage/{fileName}"
    lidarNumpyFullPath = f'{inputfolerPath}/lidarNumpyImage/{fileName[:-4]}.npy'
    lidarLabelFullPath = f"{inputfolerPath}/lidarLabelColor/{fileName}"
    lidarCloudFullPath = f"{inputfolerPath}/lidarPointCloud/{fileName[:-4]}.npy"
    lidarLabelIDFullPath = f"{inputfolerPath}/lidarLabelID/{fileName}"
    lidarSegmentedFullPath = f"{outputFolder}/lidarPred/{fileName}"
    
    lidarSegmented = cv2.imread(lidarSegmentedFullPath)
    lidarLabelID = cv2.imread(lidarLabelIDFullPath, -1)
    lidarLabel = cv2.imread(lidarLabelFullPath)
    predClassNumpy = np.array(classImage, dtype=np.int)
    cv2.imwrite("class.png", predClassNumpy*10)
    #rgbLabledLidar, visualised = lidarRGBcombine(lidarCloudFullPath,rgb_img , lidarImageFullPath, lidarNumpyFullPath , lidarLabelFullPath)
    rgbLabledLidar, visualised = lidarRGBcombine(lidarCloudFullPath,rgb_img , lidarImageFullPath, lidarNumpyFullPath , lidarLabelFullPath)
    cv2.imwrite(f"{outputFolder}/visualisert/{fileName[:-4]}.png",visualised)
    cv2.imwrite(f"{outputFolder}/labelFromRGB/{fileName[:-4]}.png",rgbLabledLidar)

    #mapper = MaskFormerSemanticDatasetMapperTest(cfg, True, numpyData=True)
    #dataLoader = build_detection_test_loader(cfg, "ffi_test_stuffonly", mapper=mapper)

    #predictor = DefaultPredictor(cfg)
    #evaluator_test = SemSegEvaluator()
    #evaluator_test.process(inputs, outputs)
    #evaluator = DatasetEvaluators(evaluator)

    #print(inference_on_dataset(predictor.model, dataLoader, evaluator_test))
    rgbLabledLidar = rgbLabledLidar.astype(np.int64)
    width = 2048
    height = 64*4
    dim = (width, height)
    rgbLabledLidarTaller = cv2.resize(rgbLabledLidar, dim, interpolation = cv2.INTER_NEAREST)
    



    #print(lidarLabelSmaller[:,:,0].shape)
    #print(rgbLabledLidar[:,:,0].shape)


    gt = lidarLabel
    pred = rgbLabledLidarTaller

    #gt[:,:] = 4
    #pred[:,:] = 4

    print("pred", pred.shape)
    print("gt", gt.shape)
    print("matrix", _conf_matrix.shape)
    print("classes", _num_classes)
    gt[(pred == 0).all(2)] = (0,0,0)
    lidarSegmented[(pred == 0).all(2)] = (0,0,0)
    cv2.imwrite("gt.png", gt)
    cv2.imwrite("pred.png", pred)

    cv2.imwrite(f"{outputFolder}/gt/{fileName[:-4]}.png",gt)
    cv2.imwrite(f"{outputFolder}/pred/{fileName[:-4]}.png",pred)
    cv2.imwrite(f"{outputFolder}/lidarPredCut/{fileName[:-4]}.png",lidarSegmented)

    

    


    #print(lidarLabelSmaller.reshape(-1).shape)
    #print(rgbLabledLidar.reshape(-1).shape)
    '''
    countTing = np.bincount(
        (_num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
        minlength=_conf_matrix.size,
    )
    print("countTing", countTing.shape)

    _conf_matrix += np.bincount(
        (_num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
        minlength=_conf_matrix.size,
    ).reshape(_conf_matrix.shape)
    '''
    

    #lidarRGBcombine()
#print(_conf_matrix)

#rgbFullPath = '/home/potetsos/lagrinatorn/master/ffiRGBdatasetAndLidar/train/image/plainsDrive_frame01947.png'
#lidarFullPath = "/home/potetsos/lagrinatorn/master/ffiRGBdatasetAndLidar/train/lidarImage/plains_drive04550.png"
#lidarNumpyFullPath = '/home/potetsos/skule/2022/masterCode/masterToft/data/lidarNumpy/plains_drive/plains_drive04550.npy'
#lidarLabelFullPath = "/home/potetsos/lagrinatorn/master/cvatLablet/combinedDataset/train/colorSegment/lidar/4.png"
#lidarCloudFullPath = "/home/potetsos/lagrinatorn/master/ffiRGBdatasetAndLidar/train/lidarNumpy/plains_drive04550.npy"


#lidarRGBcombine(lidarNumpyFullPath,rgbFullPath , lidarFullPath,lidarCloudFullPath , lidarLabelFullPath)


#inferance.writeDatasetToFile()

'''
torch.Size([1, 192, 80, 640])
torch.Size([1, 384, 40, 320])
torch.Size([1, 768, 20, 160])
torch.Size([1, 1536, 10, 80])


torch.Size([1, 192, 160, 216])
torch.Size([1, 384, 80, 108])
torch.Size([1, 768, 40, 54])
torch.Size([1, 1536, 20, 27])
'''

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

#res = evaluate()
#print(res)