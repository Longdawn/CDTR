"""
SGQ Stage-1: short glyph-aware query CE.

This config keeps the baseline architecture and decoding path unchanged. During
CTC finetuning, len<=2 samples receive an auxiliary query-level CE term: each GT
character is assigned to the currently strongest unused query for that character.
The loss is train-only and can be evaluated with or without decode-time CTC
blank/nonblank calibration.
"""

from config.MTH1000_dtlr import *


short_gt_ce_loss_coef = 0.02
short_gt_ce_max_len = 2
short_gt_ce_gamma = 1.0
