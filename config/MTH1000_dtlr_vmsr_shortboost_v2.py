from config.MTH1000_dtlr_vmsr_probe import *


# Combine vertical MSR with the milder short-sample reweighting that improved
# CER in shortboost_v2. Length-aware rules only affect short near-square crops;
# all other samples keep the VMSR bucket policy from MTH1000_dtlr_vmsr_probe.
mth1000_length_balance = True
mth1000_length_sample_weights = {
    "1": 3.0,
    "2": 2.0,
    "3-5": 1.5,
    "6-10": 1.0,
    "11+": 1.0,
}

mth1000_use_length_msr = True
mth1000_train_length_msr_rules = [
    (1, 1, 1.0, 2.2, [800]),
    (2, 2, 1.0, 2.2, [768, 800]),
    (3, 5, 1.25, 2.0, [768]),
]
mth1000_eval_length_msr_rules = []
