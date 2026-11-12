"""
MTH1000-DTLR CTC finetuning config with vertical MSR resizing.

This keeps the original MTH1000 CTC setup and only changes the resize policy:
near-square/short vertical crops use smaller short-side sizes, while long
vertical columns retain the original high-resolution behavior.
"""

from config.MTH1000_dtlr import *


mth1000_use_msr = True

# Buckets are keyed by vertical aspect ratio H/W.
# Conservative MSR: keep near-square/very-short crops at high resolution because
# the pretrained checkpoint otherwise collapses many one-character samples to
# blank. Only middle aspect-ratio crops are mildly down-scaled.
mth1000_msr_train_buckets = [
    (1.5, [736, 768, 800]),
    (3.0, [704, 736, 768]),
    (6.0, [704, 736, 768]),
    (1.0e9, [768, 800]),
]

# Validation/test must be deterministic.
mth1000_msr_eval_buckets = [
    (1.5, 800),
    (3.0, 768),
    (6.0, 768),
    (1.0e9, 800),
]
