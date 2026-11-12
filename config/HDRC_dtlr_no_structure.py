"""
HDRC no-localization CTC control.

This is a conservative lower-bound control for testing whether the
localization-supervised query initialization matters beyond MTHv2.  It trains
the line-level CTC path directly on HDRC without resuming from a character-box
localization checkpoint.  Do not compare it as a pure causal estimate of
character-box supervision; use it as a no-localization baseline under the same
single-column recognition protocol.
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
