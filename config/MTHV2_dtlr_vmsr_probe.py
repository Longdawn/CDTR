"""
MTHv2-combo vertical MSR probe.

This config keeps the MTHv2 query-to-CTC model unchanged and only enables the
aspect-ratio-aware resize policy already available in the MTH1000/MTHCombo data
pipeline.  It is intended as a low-risk SVTRv2-inspired scale probe.
"""

from config.MTHV2_dtlr import *


mth1000_use_msr = True
mth1000_use_length_msr = False

mth1000_msr_train_buckets = [
    (1.5, [768, 800], 1600),
    (3.0, [640, 704], 1800),
    (6.0, [480, 544], 2200),
    (8.0, [384, 448], 2200),
    (12.0, [320, 384], 2400),
    (1.0e9, [288, 320], 2400),
]

mth1000_msr_eval_buckets = [
    (1.5, 800, 1600),
    (3.0, 704, 1800),
    (6.0, 512, 2200),
    (8.0, 448, 2200),
    (12.0, 384, 2400),
    (1.0e9, 320, 2400),
]
