# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
import detectron2.data.transforms as T

import cv2

__all__ = ["MaskFormerSemanticDatasetMapperTest"]


class MaskFormerSemanticDatasetMapperTest:
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
        numpyData = True,
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
        self.numpyData = numpyData,
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
        #assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        #image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        #print(dataset_dict["file_name"])
        #imageNumpy = np.load()
        ##filePaths = dataset_dict["file_name"].split("/")
        ##filePaths[2] = "imagesNumpy"
        ##numpyPath = '/'.join(filePaths)[:-4] + ".npy"
        
        #numpyImage = np.load(numpyPath)
        #print("numpy?", type(self.numpyData[0]))
        #print("numpy?", self.numpyData[0])
        if (self.numpyData[0] == True):
        #    print("bad")
            #image = np.load(dataset_dict["file_name"])
            numpyImage = np.load(dataset_dict["file_name"])
            numpyImage = numpyImage.astype('float32')
            numpyImage = numpyImage = numpyImage[:,:,2:3]
            #image1 = numpyImage[:,:,0]
            #image2 = numpyImage[:,:,2:4]
            #image = np.dstack((image1, image2))
            image = numpyImage

        else:
        #    print("fint")
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

        #image = numpyImage

        height, width = image.shape[:2]
        #image = self.aug.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        dataset_dict["sem_seg"] = sem_seg_gt.long()

        image_shape = (image.shape[-2], image.shape[-1])
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
                #print(masks[0].shape)
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        
        dataset_dict['image'] = image
        dataset_dict['height'] = height
        dataset_dict['width'] = width
        #test_input_dict = {"image": image, "height": height, "width": width}

        
        return dataset_dict
