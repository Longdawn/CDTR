"""
MTH1000 short-sample-boost finetuning config.

This keeps the current best MTH1000 vertical setup, then adds two training-side
interventions for the known short-sample failure mode:

1. length-balanced sampling so 1-2 char crops are seen more often
2. train-only length-aware MSR rules so near-square short crops are resized more
   conservatively than normal long columns

Evaluation stays deployment-safe: no GT-length-dependent resize at eval time.
"""

from config.MTH1000_dtlr_msr_v2 import *


mth1000_length_balance = True
mth1000_length_sample_weights = {
    "1": 4.0,
    "2": 3.0,
    "3-5": 2.0,
    "6-10": 1.0,
    "11+": 1.0,
}

mth1000_use_length_msr = True

# Keep the existing MSR-v2 fallback policy for all samples.
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

# Only train uses GT-length-aware overrides.
# Format: (min_len, max_len, min_ratio_exclusive, max_ratio_inclusive, sizes)
mth1000_train_length_msr_rules = [
    (1, 1, 1.0, 2.2, [800]),
    (2, 2, 1.0, 2.2, [768, 800]),
    (3, 5, 1.25, 2.0, [768]),
]
mth1000_eval_length_msr_rules = []
