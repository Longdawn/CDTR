_base_ = ["Chinese.py"]

# Stage-1 detection pretraining on glyph-bank based ancient vertical synthetic data.
# The dataset uses real MTH glyph crops where available and font fallback for
# characters not covered by the glyph bank.

num_classes = 6700

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
batch_size = 2

# 89,999 train images / batch 2 ~= 45k optimizer steps per epoch.
# Keep the config conservative; override max_iterations/epochs from CLI for probes.
epochs = 3
lr_drop = 2
save_checkpoint_interval = 1
max_iterations = 30000

mode_chr = True

mth1000_root = "synth_vertical_ancient_100k_dtlr"
mth1000_raw_root = "SynthVerticalAncient100k"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = True

direction_source = "label"
forced_direction = "vertical"
mth1000_filter_direction = "vertical"

data_aug_scales = [512]
data_aug_max_size = 1200
