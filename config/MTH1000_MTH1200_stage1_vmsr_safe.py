_base_ = ["Chinese.py"]

# Stage-1 detection training on real vertical single-column data from
# TKHMTH2200/MTH1000 + TKHMTH2200/MTH1200, using real character boxes.
num_classes = 6700

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4

# Safe VMSR caps keep batch_size=2 feasible on a 24GB GPU while matching the
# previous stage-1 setting more closely than batch_size=1.
batch_size = 2

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

# Safe Vertical MSR for full overnight stage-1 detection pretraining.
mth1000_use_msr = True
mth1000_use_length_msr = False

mth1000_msr_train_buckets = [
    (1.5, [736, 768], 1500),
    (3.0, [608, 672], 1700),
    (6.0, [448, 512], 2000),
    (8.0, [352, 416], 2000),
    (12.0, [288, 352], 2200),
    (1.0e9, [256, 320], 2200),
]

mth1000_msr_eval_buckets = [
    (1.5, 768, 1500),
    (3.0, 672, 1700),
    (6.0, 512, 2000),
    (8.0, 416, 2000),
    (12.0, 352, 2200),
    (1.0e9, 320, 2200),
]
