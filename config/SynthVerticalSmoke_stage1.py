_base_ = ["Chinese.py"]

# Smoke-stage detection training on upright vertical synthetic text.
# This config is intentionally small; it validates data format and box learning
# before scaling the synthetic set to 30k/100k.

num_classes = 1160

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
batch_size = 2

epochs = 2
lr_drop = 2
save_checkpoint_interval = 1
max_iterations = 1000
mode_chr = True

mth1000_root = "synth_vertical_smoke_dtlr"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = True
mth1000_raw_root = "SynthVerticalSmoke"

direction_source = "label"
forced_direction = "vertical"
mth1000_filter_direction = "vertical"

data_aug_scales = [512]
data_aug_max_size = 1200
