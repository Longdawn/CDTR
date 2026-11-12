"""
MTH1000 count-guided CTC finetuning config.

This keeps the current short-sample training setup and adds a lightweight
expected-count auxiliary loss. It directly targets the observed short-sample
failure mode: CTC often predicts too few nonblank tokens, including empty
strings for 1-2 character crops.
"""

from config.MTH1000_dtlr_shortboost_v2 import *


# Small coefficient: strong enough to discourage empty/under-length outputs,
# but low enough not to dominate character classification.
ctc_count_loss_coef = 0.05

# Give 1-2 character samples more count supervision, because their dominant
# errors are deletion/empty prediction.
ctc_count_loss_short_weight = 3.0

