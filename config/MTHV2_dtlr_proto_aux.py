"""
MTHv2 query glyph-prototype auxiliary loss probe.

This enables a train-time prototype branch on decoder query features, but keeps
prototype fusion disabled for the CTC logits. The branch therefore regularizes
query character features without injecting randomly initialized prototype
scores into the recognition path at inference.
"""

from config.MTHV2_dtlr import *


use_glyph_prototype_head = True
glyph_proto_dim = 256
glyph_proto_fuse_coef = 0.0
glyph_proto_temperature = 1.0
glyph_proto_trainable = True

glyph_proto_aux_loss_coef = 0.01
glyph_proto_aux_max_len = 0
glyph_proto_aux_gamma = 0.0
