"""
MTHv2-combo length-balanced sampling probe.

This is a conservative training-only sampler change for screening whether
short-column distribution shift is a useful lever on the MTHv2 CTC finetuning
path.  It does not change the model or evaluation pipeline.
"""

from config.MTHV2_dtlr import *


mth1000_length_balance = True
mth1000_length_sample_weights = {
    "1": 3.0,
    "2": 2.0,
    "3-5": 1.5,
    "6-10": 1.0,
    "11+": 1.0,
}
