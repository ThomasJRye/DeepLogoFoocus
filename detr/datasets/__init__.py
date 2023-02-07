# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'flickr_logos_27':
        from .flickr_logos_27 import build as build_flickr_logos_27
        return build_flickr_logos_27(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided root path {root} does not exist'
    train_json = 'flickr_logos_27_train.json'
    test_json = 'flickr_logos_27_test.json'
    PATHS = {
        "train": (root / 'flickr_logos_27_dataset_images', root / train_json),
        "val": (root / 'flickr_logos_27_dataset_images', root / test_json),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset