# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
#from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
import detectron2.data.transforms as T

import augmentation as aug

import cv2

__all__ = ["MaskFormerSemanticDatasetMapperCombined"]


class MaskFormerSemanticDatasetMapperCombined:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        print(self.tfm_gens)
        del self.tfm_gens[2]
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.aug = T.ResizeShortestEdge([640, 640], 2560)

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        self.counter = 0

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        self.counter +=1
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        ###image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        #print(dataset_dict["file_name"])
        #imageNumpy = np.load()
        ###filePaths = dataset_dict["file_name"].split("/")
        ###filePaths[2] = "imagesNumpy"
        ###numpyPath = '/'.join(filePaths)[:-4] + ".npy"
        #print("bra", dataset_dict["file_name"])
        numpyImage = np.load(dataset_dict["file_name"])
        rgbImage =  utils.read_image(dataset_dict["rgb_file_name"], format=self.img_format)
        #print(rgbImage.shape)
        
        #print("bra2", numpyImage.shape)
        image = numpyImage
        #print(numpyImage.shape)
        #print(numpyPath)
        #print(filePaths)
        #print(image.shape)
        #print(type(image))
        #print("datafilename", dataset_dict["file_name"])
        #imageLidar = utils.read_image("/lhome/asbjotof/asbjotof/2022/masterToft/data/incoming/altTilNo/images/plains_drive04300.png", format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        #utils.check_image_size(dataset_dict, imageLidar)

        #print(imageLidar)
        #print(image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            #print("datasetdict filename", dataset_dict['sem_seg_file_name'])
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
            #print("semsegget", sem_seg_gt)
            #print(sem_seg_gt.shape)
            #cv2.imwrite(f"testImageFor/{self.counter}.png", sem_seg_gt*10)

            
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )
        
        #print("for", image.shape)
        #cv2.imwrite(f"testImageFor/{self.counter}.png", image)
        #cv2.imwrite(f"testSemSegFor/{self.counter}.png", sem_seg_gt*10)
        #print("for sem seg shape", sem_seg_gt.shape)
        #print("for image shape", image.shape)
        #dim = (512, 512)
        #image = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
        #sem_seg_gt = cv2.resize(sem_seg_gt, dim, interpolation = cv2.INTER_NEAREST)
        #print("mitten", image.shape)
        #cv2.imwrite(f"testImageMidtten/{self.counter}.png", image)
        #aug_input = T.AugInput(image,rgbImage, sem_seg=sem_seg_gt)
        #print("lidar for", image.shape)
        #print("sem seg for", sem_seg_gt.shape)
        #print("rgb for", rgbImage.shape)

        combinedImage = np.dstack((image, rgbImage))
        #print("combined for", combinedImage.shape)

        aug_input = T.AugInput(combinedImage,rgbImage=rgbImage, sem_seg=sem_seg_gt)
        
        
        #aug_rgb = 
        #print(self.tfm_gens[0])
        #print("for transform")
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        #print("rett etter 2")
        #print("etter transform")
        combinedImage = aug_input.image
        sem_seg_gt = aug_input.sem_seg
        #rgb_img = aug_input.rgb_img
        #print("combined etter", combinedImage.shape)
        image = combinedImage[:,:,0:4]
        rgb_img = combinedImage[:,:,4:7]
        #print("lidar etter", image.shape)
        #print("sem seg etter", sem_seg_gt.shape)
        #print("rgb etter", rgb_img.shape)
        
        
        #print("\n\n\n\n")

        #print("etter image", image.shape)
        #print("etter semseg", sem_seg_gt.shape)

        #cv2.imwrite(f"testImageFor/{self.counter}.png", image)
        #print("etter", image.shape)
        #cv2.imwrite(f"testSemSeg/{self.counter}.png", sem_seg_gt)  

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        rgb_img = torch.as_tensor(np.ascontiguousarray(rgb_img.transpose(2, 0, 1)))

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            rgb_img = F.pad(rgb_img, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        
        #imageLidar = self.aug.get_transform(imageLidar).apply_image(imageLidar)
        #imageLidar = torch.as_tensor(imageLidar.astype("float32").transpose(2, 0, 1))
        

        dataset_dict["image"] = image
        dataset_dict["rgbImage"] = rgb_img
        #dataset_dict["image2"] = imageLidar
        #print("burde funka", dataset_dict["rgbImage"].shape)
        #cv2.imwrite(f"testImage/{self.counter}.png", image)

        #print("styund", image.shape)

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        #print("dataMapped", dataset_dict)
        #print("dataMapped image", dataset_dict["image"].shape)
        #print("dataMapped semseg", dataset_dict["sem_seg"].shape)

        #images = ImageList.from_tensors(images, self.size_divisibility)

        #bilde = dataset_dict["image"].cpu().detach().numpy()
        #print("bilde", bilde.shape)
        #reshapedBilde = bilde.transpose(1, 2, 0)

        #0 1 2
        #2 0 1

        #1 2 0
        #print("reshaped", reshapedBilde.shape)
        #print(reshapedBilde)

        #reshapedBilde = bilde.reshape(640, 640, 4)

        #cv2.imwrite(f"testImageEtter/{self.counter}.png", bilde)

        return dataset_dict

