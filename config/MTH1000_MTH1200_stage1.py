_base_ = ["Chinese.py"]

# Stage-1 detection training on real vertical single-column data from
# TKHMTH2200/MTH1000 + TKHMTH2200/MTH1200.
# Compared with the paper's synthetic stage-1, this keeps the same detection
# objective but replaces generated lines with real character-box supervision.

num_classes = 6700

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
batch_size = 2

# 65,388 train lines / batch 2 ~= 32,694 iters per epoch.
# 7 epochs is ~= 228,858 loader iterations, close to the paper's 225k budget.
epochs = 7
lr_drop = 6
save_checkpoint_interval = 1

max_iterations = 10000
mode_chr = True

mth_combo_roots = [
    "tkhmth2200_mth1200_dtlr",
    "tkhmth2200_mth1000_dtlr",
]
mth_combo_raw_roots = [
    "TKHMTH2200/MTH1200",
    "TKHMTH2200/MTH1000",
]

mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = True

direction_source = "label"
forced_direction = "vertical"
mth1000_filter_direction = "vertical"

data_aug_scales = [768]
data_aug_max_size = 1600
