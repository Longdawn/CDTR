"""
MTH1000-DTLR length-aware MSR diagnostic config.

This uses GT text length during resize, so it is an oracle diagnostic rather
than a deployment-safe inference policy. It tests whether preserving one-char
samples at 800 while scaling two-char samples in 1.5 < H/W <= 2.0 improves the
short-line bottleneck.
"""

from config.MTH1000_dtlr import *


mth1000_use_msr = True
mth1000_use_length_msr = True

# Fallback keeps the original resize behavior.
mth1000_msr_train_buckets = [
    (1.0e9, [768, 800]),
]
mth1000_msr_eval_buckets = [
    (1.0e9, 800),
]

# (min_len, max_len, min_ratio_exclusive, max_ratio_inclusive, sizes)
mth1000_train_length_msr_rules = [
    (2, 2, 1.5, 2.0, [736, 768]),
]
mth1000_eval_length_msr_rules = [
    (2, 2, 1.5, 2.0, 768),
]
