"""Paper-facing SAQT model entry point.

The current implementation reuses the repository's query-transformer backbone
for checkpoint compatibility, while exposing `modelname = "saqt"` for new
paper-facing configs and launch commands.
"""

from .saqt import build_saqt

__all__ = ["build_saqt"]
