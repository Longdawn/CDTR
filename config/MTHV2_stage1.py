_base_ = ["Chinese.py"]

# Stage-1 detection/localization-aware pretraining on full MTHv2:
# TKHMTH2200/MTH1000 + TKHMTH2200/MTH1200 + TKHMTH2200/TKH.
# This is an experiment-only config and does not modify the trusted
# MTH1000+MTH1200 stage-1 baseline.

num_classes = 6727

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
batch_size = 2

# Full MTHv2 train lines:
#   MTH1000 29423 + MTH1200 35965 + TKH 18788 = 84176
# batch_size=2 => about 42088 loader iterations per epoch.
# Six epochs gives about 252k optimization steps, close to the existing
# real-stage1 budget while covering the added TKH subset.
epochs = 6
lr_drop = 5
save_checkpoint_interval = 1

# main_synthetic.py stage-1 training has historically ignored max_iterations;
# the effective budget is len(train_loader) * epochs.
max_iterations = 10000

# In the current codebase, mode_chr=True routes training through CTC loss.
# Stage-1 pretraining must use DETR detection losses on char boxes instead.
mode_chr = False

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
mth1000_use_char_boxes = True

direction_source = "label"
forced_direction = "vertical"
mth1000_filter_direction = "vertical"

data_aug_scales = [768]
data_aug_max_size = 1600
