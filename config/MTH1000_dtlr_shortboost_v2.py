"""
MTH1000 short-sample-boost v2.

Compared with the first shortboost run, this version keeps the same training
policy but reduces the short-sample oversampling strength to avoid the late
blank-drift failure mode observed after epoch 8-9.
"""

from config.MTH1000_dtlr_shortboost import *


# Milder length-balanced sampling.
mth1000_length_sample_weights = {
    "1": 3.0,
    "2": 2.0,
    "3-5": 1.5,
    "6-10": 1.0,
    "11+": 1.0,
}
