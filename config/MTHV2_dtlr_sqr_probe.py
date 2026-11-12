"""
MTHv2 probe for sorted query sequence refinement.

This keeps the MTHv2 recognition protocol unchanged and only enables a
lightweight residual 1D refiner over decoder query features before the final
classification head.
"""

from config.MTHV2_dtlr import *


use_query_sequence_refiner = True
query_sequence_refiner_layers = 1
query_sequence_refiner_kernel = 3
query_sequence_refiner_dropout = 0.0
