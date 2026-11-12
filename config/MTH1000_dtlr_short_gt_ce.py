"""
MTH1000 short-sample GT character CE finetuning config.

This targets the diagnosed len<=2 failure mode more directly than count-guided
or margin presence losses: for each short-sample GT character, select the query
that currently gives that character the highest probability and apply a small
NLL penalty to make the character explicitly appear in the query logits.
"""

from config.MTH1000_dtlr import *


short_gt_ce_loss_coef = 0.01
short_gt_ce_max_len = 2
short_gt_ce_gamma = 1.0

