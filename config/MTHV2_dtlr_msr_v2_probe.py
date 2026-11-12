"""
MTHv2-combo conservative MSR-v2 probe.

This only changes the near-square vertical crop band that was useful in earlier
MTH1000 diagnostics.  Other aspect ratios keep the baseline 800 short-side
evaluation policy, so this probe stays close to the original runtime.
"""

from config.MTHV2_dtlr import *


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
