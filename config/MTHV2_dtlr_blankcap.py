"""
MTHv2 CTC blank-cap probe.

This keeps the recognition model unchanged and only applies a mild cap to the
constructed CTC blank probability during training/evaluation. The goal is to
test whether a lightweight anti-collapse probability constraint improves the
same-path continuation baseline without changing decode-time calibration.
"""

from config.MTHV2_dtlr import *


ctc_blank_max = 0.995
