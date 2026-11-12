"""
MTH1000 shortboost + short GT-presence v1.

This is the first paper-rescue real-mainline config:
- keeps the trusted real-data finetuning path
- uses the milder short-sample length-balanced sampling from shortboost_v2
- adds short GT-presence loss for len<=2 samples

It does not depend on synthetic pretraining changes.
"""

from config.MTH1000_dtlr_shortboost_v2 import *


short_gt_presence_loss_coef = 0.02
short_gt_presence_max_len = 2
short_gt_presence_margin = 0.0
