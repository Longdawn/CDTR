"""
Horizontal comparison config built from the vertical MTH1000-DTLR baseline.

This config targets a rotated copy of mth1000_dtlr where every original
vertical line image has been rotated 90 degrees clockwise into a
left-to-right layout. It is intended only for controlled comparison
experiments and does not modify the existing vertical pipeline.
"""

from config.MTH1000_dtlr import *


mth1000_root = "mth1000_dtlr_rot90"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = False

direction_source = "label"
forced_direction = "horizontal"
mth1000_filter_direction = "horizontal"
