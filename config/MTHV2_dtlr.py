"""
MTHv2-style joint finetuning config.

This combines the three processed vertical-line subsets that originally belong
to the larger TKHMTH2200/MTHv2 family:
- MTH1000
- MTH1200
- TKH

Use this for CTC head reconstruction and full finetuning on the combined line
recognition dataset. It intentionally does not modify the trusted single-dataset
configs.
"""

from config.Chinese import *


# The real stage-1 checkpoint was trained with the MTH1000+MTH1200 charset size.
# During head reconstruction, --new_class_embedding resizes the classifier to
# the actual MTHCombo union charset.
num_classes = 6700


mth_combo_roots = [
    "tkhmth2200_mth1000_dtlr",
    "tkhmth2200_mth1200_dtlr",
    "tkhmth2200_tkh_dtlr",
]

mth_combo_raw_roots = [
    "TKHMTH2200/MTH1000",
    "TKHMTH2200/MTH1200",
    "TKHMTH2200/TKH",
]

mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = False


# Real-data finetuning stays CTC-only and focuses on vertical line recognition.
mode_chr = True
use_direction_head = False
direction_loss_coef = 0.0
decode_by_pred_direction = False
direction_source = "rule"
forced_direction = "vertical"
mth1000_filter_direction = "vertical"

