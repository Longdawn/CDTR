"""
Paper-facing SAQT config for HDRC line-level recognition.

The config binds the MTH1000-style loader to processed HDRC single-column data.
Source checkpoints, charset mapping flags, and run-specific training budgets are
still supplied by the launch command so old checkpoints remain reproducible.
"""

from config.MTH1000_dtlr import *


modelname = "saqt"

mth1000_root = "hdrc_dtlr"
mth1000_raw_root = "HDRC"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = False

forced_direction = "vertical"
mth1000_filter_direction = "vertical"
