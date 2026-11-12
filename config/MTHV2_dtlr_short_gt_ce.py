"""
MTHv2 short-sample GT character CE finetuning probe.

This reuses the existing query-level short GT CE hook. For len<=2 samples,
each GT character selects the query with the highest current probability for
that character and receives a small NLL penalty. The baseline MTHv2 config is
left unchanged.
"""

from config.MTHV2_dtlr import *


short_gt_ce_loss_coef = 0.01
short_gt_ce_max_len = 2
short_gt_ce_gamma = 1.0
