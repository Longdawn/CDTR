"""
MTHv2 short GT-presence CTC finetuning probe.

This is a lightweight, DCTC-inspired alignment regularization probe for the
query-to-CTC path.  It only adds a short-sample auxiliary loss: for len<=2
samples, every ground-truth character should beat blank at least once among the
sorted queries.  The trusted MTHv2 config remains unchanged.
"""

from config.MTHV2_dtlr import *


short_gt_presence_loss_coef = 0.02
short_gt_presence_max_len = 2
short_gt_presence_margin = 0.0
