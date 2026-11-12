from config.MTH1000_MTH1200_stage1 import *


# Vertical MSR for stage-1 detection pretraining on real character boxes.
# Unlike the conservative MSR probe, this separates the dominant vertical
# aspect-ratio bands observed in MTH1000/MTH1200 and gives long columns a
# larger height cap while keeping aspect ratio intact.
mth1000_use_msr = True
mth1000_use_length_msr = False

mth1000_msr_train_buckets = [
    (1.5, [768, 800], 1600),
    (3.0, [640, 704], 1800),
    (6.0, [480, 544], 2200),
    (8.0, [384, 448], 2200),
    (12.0, [320, 384], 2400),
    (1.0e9, [288, 320], 2400),
]

mth1000_msr_eval_buckets = [
    (1.5, 800, 1600),
    (3.0, 704, 1800),
    (6.0, 512, 2200),
    (8.0, 448, 2200),
    (12.0, 384, 2400),
    (1.0e9, 320, 2400),
]
