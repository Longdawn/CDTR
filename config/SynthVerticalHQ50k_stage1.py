_base_ = ["Chinese.py"]

# Stage-1 detection pretraining on higher-quality vertical synthetic MTH data.
# This dataset covers the joint MTH1000+MTH1200 charset and mixes real text
# snippets, multiple CJK fonts, short-line oversampling, blur/erase, and box
# jitter. Use this to test whether synthetic pretraining can produce real-domain
# query activations before any real character-box adaptation.

num_classes = 6700

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
batch_size = 2

# 44,999 train images / batch 2 ~= 22,499 optimizer steps per epoch.
# 6 epochs gives ~=135k optimizer steps. If detection diagnostics are still
# improving, extend from checkpoint rather than starting a new log.
epochs = 6
lr_drop = 5
save_checkpoint_interval = 1
max_iterations = 30000
mode_chr = True

mth1000_root = "synth_vertical_hq50k_dtlr"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = True
mth1000_raw_root = "SynthVerticalHQ50k"

direction_source = "label"
forced_direction = "vertical"
mth1000_filter_direction = "vertical"

data_aug_scales = [512]
data_aug_max_size = 1200
