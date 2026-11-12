"""
HDRC full-finetuning ablation with a conservative CTC expected-count loss.

This mirrors the matched HDRC qbudget-localization-query full-finetuning
protocol and only adds the count-consistency term that was positive on MTHv2.
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

ctc_count_loss_coef = 0.01
ctc_count_loss_short_weight = 3.0
