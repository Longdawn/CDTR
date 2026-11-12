# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'synthetic_line_OCR_general':
        from .synthetic_lines_general import build_synthetic_line_OCR_general
        return build_synthetic_line_OCR_general(image_set, args)
    if args.dataset_file == 'mth1000':
        from .MTH1000 import build_mth1000
        return build_mth1000(image_set, args)
    if args.dataset_file == 'mth_combo':
        from .MTHCombo import build_mth_combo
        return build_mth_combo(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
