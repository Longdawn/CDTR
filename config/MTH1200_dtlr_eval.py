"""
MTH1200 evaluation config for line-level DTLR finetuning checkpoints.

Use this when evaluating an existing CTC-style checkpoint on the processed
MTH1200 line dataset without changing the shared MTH1000 baseline config.
"""

from config.MTH1000_dtlr import *


mth1000_root = "tkhmth2200_mth1200_dtlr"
mth1000_raw_root = "TKHMTH2200/MTH1200"
