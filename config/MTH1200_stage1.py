_base_ = ["Chinese.py"]

# Stage-1 detection training on real vertical single-column data from TKHMTH2200/MTH1200.
# This follows the paper's first-stage spirit (box + class supervision), but uses
# real character-level annotations instead of synthetic data.

num_classes = 5298

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
batch_size = 2

# 35,965 train lines / batch 2 ~= 17,982 iters per epoch.
# 13 epochs is ~233,766 iterations, close to the paper's 225k stage-1 budget.
epochs = 13
lr_drop = 10
save_checkpoint_interval = 2

max_iterations = 10000
mode_chr = True

mth1000_root = "tkhmth2200_mth1200_dtlr"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = True
mth1000_raw_root = "TKHMTH2200/MTH1200"

direction_source = "label"
forced_direction = "vertical"
mth1000_filter_direction = "vertical"

# Keep augmentation conservative for the first verification run.
# Vertical single-column pages are much taller than the original synthetic stage-1 lines,
# so we reduce the effective image budget to avoid OOM during detection pretraining.
data_aug_scales = [768]
data_aug_max_size = 1600
