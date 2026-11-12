"""
Paper-facing SAQT config for MTHv2-combo recognition.

This aliases the current strongest MTHv2 setup: query-budget localization
pretraining followed by CTC recognition finetuning with a conservative
expected-count auxiliary term. Historical configs keep their original names for
reproducibility; new paper-facing commands should prefer this file.
"""

from config.MTHV2_dtlr_ctc_count001 import *


modelname = "saqt"
