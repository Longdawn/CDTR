"""
Clean MTH1000-DTLR finetuning config built from the original Chinese family.

Design choices:
- inherit only from original repo configs
- use line-level supervision (paper-style real-data finetuning)
- restrict training to vertical lines from mth1000_dtlr
- keep only task-specific overrides in config
- let training policy be chosen explicitly from the launch command
"""

from config.Chinese import *


# Explicitly bind this config to the processed MTH1000-DTLR line dataset.
mth1000_root = "mth1000_dtlr"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_raw_root = "MTH1000"
mth1000_use_char_boxes = False


# Real-data finetuning stays CTC-only and focuses on vertical lines.
mode_chr = True
use_direction_head = False
direction_loss_coef = 0.0
decode_by_pred_direction = False
direction_source = "rule"
forced_direction = "vertical"
mth1000_filter_direction = "vertical"
