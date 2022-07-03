from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from detectron2.utils.file_io import PathManager
import logging
logger = logging.getLogger(__name__)

def register_combined_dataset(name, metadata, lidar_root, rgb_root, sem_seg_root):
    semantic_name = name
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg_rgb_lidar(sem_seg_root, rgb_root=rgb_root,lidar_root=lidar_root, gt_ext="png", lidar_ext="npy", rgb_ext="png"))
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        lidar_root=lidar_root,
        rgb_root=rgb_root,
        evaluator_type="sem_seg",
        ignore_label=0,
        **metadata,
    
)

def register_numpy_dataset(
    name, metadata, image_root, sem_seg_root
):
    """
    Register a "separated" version of COCO panoptic segmentation dataset named `name`.
    The annotations in this registered dataset will contain both instance annotations and
    semantic annotations, each with its own contiguous ids. Hence it's called "separated".

    It follows the setting used by the PanopticFPN paper:

    1. The instance annotations directly come from polygons in the COCO
       instances annotation task, rather than from the masks in the COCO panoptic annotations.

       The two format have small differences:
       Polygons in the instance annotations may have overlaps.
       The mask annotations are produced by labeling the overlapped polygons
       with depth ordering.

    2. The semantic annotations are converted from panoptic annotations, where
       all "things" are assigned a semantic id of 0.
       All semantic categories will therefore have ids in contiguous
       range [1, #stuff_categories].

    This function will also register a pure semantic segmentation dataset
    named ``name + '_stuffonly'``.

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images
        panoptic_json (str): path to the json panoptic annotation file
        sem_seg_root (str): directory which contains all the ground truth segmentation annotations.
        instances_json (str): path to the json instance annotation file
    """

    
    semantic_name = name
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root))
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        image_root=image_root,
        evaluator_type="sem_seg",
        ignore_label=0,
        **metadata,
    )
    

    '''
    panoptic_name = name + "_separated"
    DatasetCatalog.register(
        panoptic_name,
        lambda: merge_to_panoptic(
            load_coco_json(instances_json, image_root, panoptic_name),
            load_sem_seg(sem_seg_root, image_root),
        ),
    )
    
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        sem_seg_root=sem_seg_root,
        json_file=instances_json,  # TODO rename
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        **metadata,
    )
    
    semantic_name = name + "_stuffonly"
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root))
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        image_root=image_root,
        evaluator_type="sem_seg",
        ignore_label=255,
        **metadata,
    )
    '''

def register_relllis_png_dataset(
    name, metadata, image_root, sem_seg_root
):
    """
    Register a "separated" version of COCO panoptic segmentation dataset named `name`.
    The annotations in this registered dataset will contain both instance annotations and
    semantic annotations, each with its own contiguous ids. Hence it's called "separated".

    It follows the setting used by the PanopticFPN paper:

    1. The instance annotations directly come from polygons in the COCO
       instances annotation task, rather than from the masks in the COCO panoptic annotations.

       The two format have small differences:
       Polygons in the instance annotations may have overlaps.
       The mask annotations are produced by labeling the overlapped polygons
       with depth ordering.

    2. The semantic annotations are converted from panoptic annotations, where
       all "things" are assigned a semantic id of 0.
       All semantic categories will therefore have ids in contiguous
       range [1, #stuff_categories].

    This function will also register a pure semantic segmentation dataset
    named ``name + '_stuffonly'``.

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images
        panoptic_json (str): path to the json panoptic annotation file
        sem_seg_root (str): directory which contains all the ground truth segmentation annotations.
        instances_json (str): path to the json instance annotation file
    """

    
    semantic_name = name
    #print("flott", semantic_name)
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root, image_ext="jpg", gt_ext="png"))
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        image_root=image_root,
        evaluator_type="sem_seg",
        ignore_label=0,
        **metadata,
    )

def register_ffi_png_dataset(
    name, metadata, image_root, sem_seg_root
):

    semantic_name = name
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root, image_ext="png", gt_ext="png"))
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        image_root=image_root,
        evaluator_type="sem_seg",
        ignore_label=0,
        **metadata,
    )

def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="npy"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def load_sem_seg_rgb_lidar(gt_root, rgb_root, lidar_root, gt_ext="png", rgb_ext="png", lidar_ext="npy"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(lidar_root, f) for f in PathManager.ls(lidar_root) if f.endswith(lidar_ext)),
        key=lambda file_path: file2id(lidar_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                lidar_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(lidar_root)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(lidar_root, f + lidar_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), lidar_root)
    )




    print("rgb_root", rgb_root)
    rgb_files = sorted(
        (os.path.join(rgb_root, f) for f in PathManager.ls(rgb_root) if f.endswith(rgb_ext)),
        key=lambda file_path: file2id(rgb_root, file_path),
    )

    print("rgb_files for", rgb_files)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(rgb_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                rgb_root, gt_root, len(rgb_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(rgb_root)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        rgb_files = [os.path.join(rgb_root, f + rgb_ext) for f in intersect]
        #gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(rgb_files), rgb_ext)
    )
    
    print("input_files", input_files)
    print("rgb_files", rgb_files)






    dataset_dicts = []
    for (img_path, gt_path, rgb_path) in zip(input_files, gt_files, rgb_files):
        record = {}
        record["file_name"] = img_path
        record["rgb_file_name"] = rgb_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts

'''
for sem seg shape (256, 2048)
for image shape (4, 256, 2048)
ResizeShortestEdge(short_edge_length=..., max_size=2560, sample_style='choice')

for sem seg shape (256, 2048)
for image shape (256, 2048, 4)
ResizeShortestEdge(short_edge_length=..., max_size=2560, sample_style='choice')

'''

