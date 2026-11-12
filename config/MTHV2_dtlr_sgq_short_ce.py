"""
MTHv2 SGQ short-query CE experiment config.

Recognition-only finetuning on the MTHv2 combined line dataset. This imports
the MTHv2 baseline config unchanged and enables only the train-time short
query CE auxiliary loss for len<=2 samples.
"""

from config.MTHV2_dtlr import *


short_gt_ce_loss_coef = 0.02
short_gt_ce_max_len = 2
short_gt_ce_gamma = 1.0
