from config.MTH1000_dtlr_vmsr_shortboost_v2 import *


# v3 keeps VMSR unchanged but softens short-sample oversampling and gives the
# 6-10 length bucket a small boost, because v2 improved len=1 and len>=11 while
# regressing on len=6-10.
mth1000_length_sample_weights = {
    "1": 2.5,
    "2": 1.8,
    "3-5": 1.3,
    "6-10": 1.2,
    "11+": 1.0,
}
