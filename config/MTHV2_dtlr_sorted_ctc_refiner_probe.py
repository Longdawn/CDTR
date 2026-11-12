"""
MTHv2 probe for sorted CTC-only query sequence refinement.

Unlike the query-index SQR probe, this branch sorts decoder query features by
the predicted vertical coordinate before applying the lightweight sequence
refiner, and uses the refined logits only in the CTC recognition path.
"""

from config.MTHV2_dtlr import *


use_sorted_ctc_refiner = True
sorted_ctc_refiner_axis = 1
sorted_ctc_refiner_layers = 1
sorted_ctc_refiner_kernel = 3
sorted_ctc_refiner_dropout = 0.0
