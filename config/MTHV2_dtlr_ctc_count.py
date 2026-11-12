"""
MTHv2 conservative CTC expected-count auxiliary loss probe.

This adds a small count-ratio loss on the CTC nonblank probability mass. It is
intended as a short-window probe for deletion/under-decoding behavior, without
changing the shared MTHv2 baseline config.
"""

from config.MTHV2_dtlr import *


ctc_count_loss_coef = 0.02
ctc_count_loss_short_weight = 3.0
