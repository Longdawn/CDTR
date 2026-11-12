"""
MTHv2 full-finetuning ablation with a conservative CTC expected-count loss.

This keeps the MTHv2 recognition setup unchanged and only adds a small
nonblank-count consistency term. The coefficient follows the stable quick
probe on 2026-06-07.
"""

from config.MTHV2_dtlr import *


ctc_count_loss_coef = 0.01
ctc_count_loss_short_weight = 3.0
