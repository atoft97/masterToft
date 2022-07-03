# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    #SemSegEvaluator,
    verify_results,
    inference_on_dataset,
)

from newSemSegEval import SemSegEvaluator

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
import sys

from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_panoptic
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from registerDataset import register_ffi_png_dataset
from mask_former_semantic_dataset_mapper_test import MaskFormerSemanticDatasetMapperTest
from registerDataset import register_numpy_dataset
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



register_numpy_dataset(name="ffi_train_stuffonly",   metadata={}, sem_seg_root="ffiLiDARdataset/train/idSegment", image_root="ffiLiDARdataset/train/numpyImage")
register_numpy_dataset(name="ffi_test_stuffonly", metadata={}, sem_seg_root="ffiLiDARdataset/test/idSegment", image_root="ffiLiDARdataset/test/numpyImage")

#register_numpy_dataset(name="ffi_train_stuffonly",   metadata={}, sem_seg_root="ffiLiDARdatasetSmaller/train/idSegment", image_root="ffiLiDARdatasetSmaller/train/numpyImage")
#register_numpy_dataset(name="ffi_test_stuffonly", metadata={}, sem_seg_root="ffiLiDARdatasetSmaller/test/idSegment", image_root="ffiLiDARdatasetSmaller/test/numpyImage")

classes = ["other"]
colors = [[0,0,0]]

for category in COCO_CATEGORIES:
    classes.append(category['name'])
    colors.append(category['color'])

#MetadataCatalog.get("ffi_train_stuffonly").set(stuff_classes=['other', 'Grass', 'CameraEdge', 'Vehicle', 'Person', 'Bush', 'Puddle', 'Building', 'Dirtroad', 'Sky', 'Large_stone', 'Forrest', 'Gravel'])
#MetadataCatalog.get("ffi_train_stuffonly").set(stuff_colors=[[255,255,255], [64,255,38], [70,70,70], [150,0,191], [255,38,38], [232,227,81], [255,179,0], [255,20,20], [191,140,0], [15,171,255], [200,200,200], [46,153,0], [180,180,180]])

MetadataCatalog.get("ffi_train_stuffonly").set(stuff_classes=classes)
MetadataCatalog.get("ffi_train_stuffonly").set(stuff_colors=colors)

MetadataCatalog.get("ffi_test_stuffonly").set(stuff_classes=classes)
MetadataCatalog.get("ffi_test_stuffonly").set(stuff_colors=colors)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        '''
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
        '''
        evaluator_list = []
        evaluator_list.append(SemSegEvaluator(dataset_name, output_folder))
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        '''
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)
        '''
                # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        # Panoptic segmentation dataset mapper
        #elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
        #    mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # DETR-style dataset mapper for COCO panoptic segmentation
        #elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
        #    mapper = DETRPanopticDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg
    """

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml")
    #cfg.merge_from_file("configs/ade20k/semantic-segmentation/swin/maskformer2_swin_small_bs16_160k.yaml")
    #cfg.merge_from_file("panopticConfig.yaml")
    #cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])
    cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
    #cfg.MODEL.WEIGHTS = 'models/model_final_6b4a3a.pkl'
    #cfg.MODEL.WEIGHTS = 'models/swinS.pkl'
    #cfg.MODEL.WEIGHTS = 'models/2batch.pth'
    cfg.MODEL.WEIGHTS = 'models/ffiLidarRange.pth'
    
    
    

    cfg.DATASETS.TRAIN = ("ffi_train_stuffonly", )
    cfg.DATASETS.TEST = ("ffi_test_stuffonly", )
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.000002#la på ein 0

    cfg.SOLVER.MAX_ITER = 29*50*3*12
    cfg.SOLVER.STEPS = []
    cfg.TEST.EVAL_PERIOD = 29*5*3*12
    #cfg.MODEL.PIXEL_MEAN = [128,128,128,128]
    #cfg.MODEL.PIXEL_STD = [58,58,58,58]

    #cfg.MODEL.PIXEL_MEAN = [128,128,128]
    #cfg.MODEL.PIXEL_STD = [58,58,58]

    #cfg.MODEL.PIXEL_MEAN = [106,52,77,143]
    #cfg.MODEL.PIXEL_STD = [83,70,116,65]

    #cfg.MODEL.PIXEL_MEAN = [106,52,1979,143]
    #cfg.MODEL.PIXEL_STD = [83,70,2981,65]

    cfg.MODEL.PIXEL_MEAN = [1979/256]
    cfg.MODEL.PIXEL_STD = [2981/256]

    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 0
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 14

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.OUTPUT_DIR = "./ffiLidarRangeTest"

    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")

    return(cfg)


#def main(args):
#    cfg = setup(args)

    '''
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()
    '''
#    trainer = Trainer(cfg)
#    trainer.resume_or_load(resume=False)
#    return trainer.train()


#if __name__ == "__main__":
#    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
#    args = default_argument_parser().parse_args()
#    print("Command Line Args:", args)
    '''
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    '''
#    launch(
#        main,
#        1,
#        num_machines=1,
#        machine_rank=0,
#        dist_url="tcp://127.0.0.1:{}".format(port),
#        args=(args,),
#    )


args = default_argument_parser().parse_args()
cfg = setup(args)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
#trainer.train()


mapper = MaskFormerSemanticDatasetMapperTest(cfg, True, numpyData=True)
dataLoader = build_detection_test_loader(cfg, "ffi_test_stuffonly", mapper=mapper)

predictor = DefaultPredictor(cfg)
evaluator_test = SemSegEvaluator(dataset_name="ffi_test_stuffonly")
#val_loader = build_detection_test_loader(cfg, dataset_name="ffi_test_stuffonly")
#print(inference_on_dataset(predictor.model, val_loader, evaluator_test))
#mapper = MaskFormerSemanticDatasetMapperTest(cfg, True)
#val_loader = build_detection_test_loader(cfg, dataset_name="ffi_test_stuffonly", mapper=mapper)

#mapper = MaskFormerSemanticDatasetMapperTest(cfg, True)
#return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

print(inference_on_dataset(predictor.model, dataLoader, evaluator_test))