"""
MTHv2 weak CTC expected-count auxiliary loss sanity check.

This matches the qbudget MTHv2 full-finetuning protocol but reduces the
expected-count coefficient from 0.01 to 0.003.  The purpose is not to tune a new
main result; it tests whether the count regularizer remains helpful with a
weaker weight that may transfer more safely to HDRC.
"""

from config.MTHV2_dtlr import *


ctc_count_loss_coef = 0.003
ctc_count_loss_short_weight = 3.0
