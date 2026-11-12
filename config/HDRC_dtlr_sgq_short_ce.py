"""
HDRC SGQ short-query CE experiment config.

Recognition-only finetuning on processed HDRC line crops. This leaves the
shared MTH1000 baseline config untouched and enables only the train-time short
query CE auxiliary loss for len<=2 samples.
"""

from config.MTH1000_dtlr import *


num_classes = 7356

mth1000_root = "hdrc_dtlr"
mth1000_raw_root = "HDRC"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = False

forced_direction = "vertical"
mth1000_filter_direction = "vertical"

short_gt_ce_loss_coef = 0.02
short_gt_ce_max_len = 2
short_gt_ce_gamma = 1.0
