"""
MTHv2 localization-guided query-to-CTC adapter experiment.

This config keeps the MTHv2 recognition setup intact and only enables the
lightweight activation-gated query-to-CTC adapter. It is intended for smoke
tests from an existing MTHv2 head-reconstruction checkpoint before any paper
claim is made.
"""

from config.MTHV2_dtlr import *


use_query_activation_head = True
query_activation_init_bias = -4.0

use_activation_gating = True
activation_gate_blank_coef = 0.10
activation_gate_nonblank_coef = 0.10

use_query_count_loss = True
query_count_loss_coef = 0.001
query_count_short_weight = 1.0
