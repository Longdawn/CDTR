"""Character-prior correction for vertically ordered CTC probabilities."""

import torch

from util.ctc_decoding import collapse_ctc


def log_character_prior(foreground_mass: torch.Tensor, training_characters: int) -> torch.Tensor:
    """Convert summed training-query probabilities to log count-scale priors."""
    mass = foreground_mass.to(dtype=torch.float64)
    if mass.ndim != 1 or mass.numel() == 0:
        raise ValueError("Expected one probability sum per character")
    if not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("Character probability sums must be finite and nonnegative")
    total = mass.sum()
    if total <= 0 or training_characters <= 0:
        raise ValueError("Training character count and foreground mass must be positive")
    return (mass / total * training_characters).clamp_min(1e-12).log().float()


def decode_with_character_prior(
    probabilities: torch.Tensor,
    log_prior: torch.Tensor,
    strength: float,
    nonblank_offset: float,
) -> list[int]:
    """Return zero-based character indices after CTC collapse and blank removal."""
    if probabilities.ndim != 2 or log_prior.ndim != 1:
        raise ValueError("Expected [queries, classes + 1] probabilities and a 1D prior")
    if log_prior.numel() == 0 or probabilities.shape[1] != log_prior.numel() + 1:
        raise ValueError("The prior must contain one value per nonblank class")
    scores = probabilities.clamp_min(1e-20).log()
    best_score, best_class = (scores[:, 1:] - strength * log_prior.to(scores)).max(-1)
    margin = best_score - scores[:, 0]
    labels = torch.where(margin + nonblank_offset > 0, best_class + 1, 0)
    return [label - 1 for label in collapse_ctc(labels.tolist())]
