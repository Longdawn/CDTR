_base_ = ["Chinese.py"]

# Pilot stage-1 detection pretraining on 30k upright vertical synthetic lines.
# Purpose: test whether short-heavy vertical synthetic char boxes improve query
# activation before scaling to the paper-level 100k synthetic set.

num_classes = 4310

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
batch_size = 2

# 26,999 train images / batch 2 ~= 13,499 optimizer steps per epoch.
# 8 epochs gives ~=108k optimizer steps, enough for a pilot without committing
# to the paper-scale 225k+ stage-1 budget.
epochs = 8
lr_drop = 7
save_checkpoint_interval = 2
max_iterations = 30000
mode_chr = True

mth1000_root = "synth_vertical_pilot30k_dtlr"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = True
mth1000_raw_root = "SynthVerticalPilot30k"

direction_source = "label"
forced_direction = "vertical"
mth1000_filter_direction = "vertical"

data_aug_scales = [512]
data_aug_max_size = 1200
