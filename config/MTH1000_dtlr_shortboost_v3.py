"""
MTH1000 short-sample-boost v3.

This is a single-variable follow-up after shortboost_presence_v1:
- keep the trusted real-data finetuning path
- remove the short GT-presence loss entirely
- increase short-sample sampling strength one notch above shortboost_v2

The goal is to isolate whether stronger sampling alone can beat the current
best real-mainline result without the extra presence term.
"""

from config.MTH1000_dtlr_shortboost_v2 import *


mth1000_length_sample_weights = {
    "1": 3.5,
    "2": 2.5,
    "3-5": 1.75,
    "6-10": 1.0,
    "11+": 1.0,
}
