"""
MTHv2 DCTC-lite alignment regularization probe.

This keeps the qbudget MTHv2 recognition model unchanged and adds a small
Viterbi-path CTC alignment loss for short samples during training only. It is
a quick probe for whether explicit best-path character-frame supervision
improves short-column deletion/under-decoding without adding inference
parameters.
"""

from config.MTHV2_dtlr import *


ctc_viterbi_loss_coef = 0.02
ctc_viterbi_loss_blank_weight = 0.0
ctc_viterbi_loss_nonblank_weight = 1.0
ctc_viterbi_loss_max_len = 2
