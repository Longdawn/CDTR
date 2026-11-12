"""
MTH1000 short GT-presence CTC finetuning config.

Motivation from diagnostics:
- Empty predictions on len<=2 samples are caused by blank beating nonblank.
- Many short errors also miss one of the ground-truth characters from the top
  candidates.

This loss is applied only to short samples. For each GT character, it requires
at least one query/time step to score that GT character above blank by a margin.
It is intentionally separate from length-balanced sampling so the effect can be
attributed cleanly.
"""

from config.MTH1000_dtlr import *


short_gt_presence_loss_coef = 0.02
short_gt_presence_max_len = 2
short_gt_presence_margin = 0.0

