"""SAQT registry wrapper.

This wrapper keeps historical checkpoints compatible with the existing
query-transformer implementation while giving new experiments a SAQT model
name at the config/registry level.
"""

from ..dino.dino import build_dino
from ..registry import MODULE_BUILD_FUNCS


@MODULE_BUILD_FUNCS.registe_with_name(module_name="saqt")
def build_saqt(args):
    return build_dino(args)
