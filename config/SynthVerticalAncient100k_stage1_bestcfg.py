_base_ = ["SynthVerticalAncient100k_stage1.py"]

# Match the best real MTH1000+MTH1200 stage-1 training geometry/config as
# closely as possible, changing only the dataset source to Ancient100k.
batch_size = 2
lr = 1e-4
lr_backbone = 1e-5
epochs = 7
lr_drop = 6
save_checkpoint_interval = 1
max_iterations = 10000

data_aug_scales = [768]
data_aug_max_size = 1600
