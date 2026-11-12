from config.MTH1000_MTH1200_stage1 import *


# Conservative MSR for stage-1 detection pretraining.
# Keep the original real-box supervision setup unchanged and only replace
# the fixed 768 resize with a narrow aspect-ratio-aware resize policy.
mth1000_use_msr = True
mth1000_use_length_msr = False

mth1000_msr_train_buckets = [
    (1.5, [768, 800]),
    (1.75, [736, 768]),
    (1.0e9, [768, 800]),
]

mth1000_msr_eval_buckets = [
    (1.5, 800),
    (1.75, 768),
    (1.0e9, 800),
]
