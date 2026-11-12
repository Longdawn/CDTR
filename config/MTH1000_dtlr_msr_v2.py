"""
MTH1000-DTLR MSR-v2 eval/training config.

Pure image-ratio version. It only changes the narrow aspect-ratio band that
benefited in diagnostics: 1.5 < H/W <= 1.75. Other samples stay at the original
800 short-side policy.
"""

from config.MTH1000_dtlr import *


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
