"""
HDRC weak CTC expected-count auxiliary loss sanity check.

This mirrors the HDRC qbudget-localization-query full-finetuning protocol but
uses a weaker expected-count coefficient than the count001 run.  Keep it only
if it avoids the HDRC count collapse while preserving useful validation
behavior.
"""

from config.MTH1000_dtlr import *


num_classes = 6727

mth1000_root = "hdrc_dtlr"
mth1000_raw_root = "HDRC"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = False

forced_direction = "vertical"
mth1000_filter_direction = "vertical"

ctc_count_loss_coef = 0.003
ctc_count_loss_short_weight = 3.0
