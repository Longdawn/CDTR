"""
MTHv2 probe combining CTC expected-count regularization and glyph-prototype
auxiliary supervision.

Both branches are train-time auxiliary losses. Prototype-logit fusion remains
disabled, so the recognition logits and inference path stay unchanged.
"""

from config.MTHV2_dtlr import *


ctc_count_loss_coef = 0.01
ctc_count_loss_short_weight = 3.0

use_glyph_prototype_head = True
glyph_proto_dim = 256
glyph_proto_fuse_coef = 0.0
glyph_proto_temperature = 1.0
glyph_proto_trainable = True

glyph_proto_aux_loss_coef = 0.01
glyph_proto_aux_max_len = 0
glyph_proto_aux_gamma = 0.0
