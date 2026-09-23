"""Fixed HDEA multi-view answer readout."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def geometric_pool(
    component_log_probabilities: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Equal-weight geometric pool of Core, E40, and E48 candidate probabilities.

    Returns normalized log probabilities, probabilities, and the zero-based
    winning candidate index.  The paper's Full configuration supplies exactly
    three rows, but the implementation validates any non-empty fixed set.
    """

    values = np.asarray(component_log_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("component log probabilities must have shape [views, choices>=2]")
    if not np.isfinite(values).all():
        raise ValueError("component log probabilities must be finite")
    scores = values.mean(axis=0)
    shifted = scores - float(scores.max())
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    normalized_log_probs = np.log(probabilities)
    return normalized_log_probs, probabilities, int(np.argmax(scores))

