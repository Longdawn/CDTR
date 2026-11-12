from config.MTH1000_dtlr import *

# evaluation.py does not support --options overrides, so keep an eval-only
# config that points to the processed TKHMTH2200/MTH1000 line dataset used by
# the current finetuning runs.
mth1000_root = "tkhmth2200_mth1000_dtlr"
mth1000_raw_root = "TKHMTH2200/MTH1000"

