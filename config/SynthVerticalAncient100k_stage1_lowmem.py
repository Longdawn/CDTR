_base_ = ["SynthVerticalAncient100k_stage1.py"]

# Low-memory probe for a single 24GB GPU. Keep this separate from the normal
# config so CLI parsing cannot accidentally turn list-valued options into strings.

batch_size = 1
lr = 5e-5
lr_backbone = 5e-6

epochs = 3
lr_drop = 2
max_iterations = 30000

data_aug_scales = [384]
data_aug_max_size = 900
