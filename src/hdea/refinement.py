"""Frozen Module-II core-preserving nested refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import retrieval_scores


@dataclass(frozen=True)
class NestedViews:
    """Nested evidence views in physical-frame coordinates."""

    e32: tuple[int, ...]
    e40: tuple[int, ...]
    e48: tuple[int, ...]
    residual_actions: tuple[tuple[int, ...], ...]
    action_values: tuple[float, ...]
    control_action: int | None
    ranked_actions: tuple[int, ...]
    fallback: str | None


def stable_softmax(values: Sequence[float]) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim != 1 or len(logits) < 2 or not np.isfinite(logits).all():
        raise ValueError("candidate logits must be a finite vector of length >= 2")
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def residual_positions(
    candidate_frame_indices: Sequence[int], core_frame_indices: Sequence[int]
) -> list[int]:
    """Return catalog positions whose physical frames are absent from the core."""

    candidates = [int(value) for value in candidate_frame_indices]
    if candidates != sorted(candidates) or len(candidates) != len(set(candidates)):
        raise ValueError("candidate frames must be unique and chronological")
    core = {int(value) for value in core_frame_indices}
    return [position for position, frame in enumerate(candidates) if frame not in core]


def action_windows(
    residual_catalog_positions: Sequence[int],
    *,
    window_count: int = 8,
    frames_per_window: int = 4,
) -> list[list[int]]:
    """Build eight disjoint temporal actions with four residual frames each."""

    positions = [int(value) for value in residual_catalog_positions]
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise ValueError("residual positions must be unique and chronological")
    if window_count < 2 or frames_per_window <= 0:
        raise ValueError("need at least two actions and one frame per action")
    if len(positions) < window_count * frames_per_window:
        raise ValueError("residual catalog cannot supply matched action windows")
    blocks = np.array_split(np.asarray(positions, dtype=np.int64), window_count)
    windows: list[list[int]] = []
    for block in blocks:
        if len(block) < frames_per_window:
            raise ValueError("a temporal partition is smaller than the action budget")
        offsets = np.linspace(0, len(block) - 1, frames_per_window).round().astype(int)
        selected = [int(block[index]) for index in offsets]
        if len(selected) != len(set(selected)):
            raise AssertionError("endpoint-inclusive action sampling produced duplicates")
        windows.append(selected)
    return windows


def posterior_pair_discrimination(
    option_scores: np.ndarray, posterior_logits: Sequence[float]
) -> float:
    """Compute sum_{a<b} pi_a*pi_b*mean_f |z_f(a)-z_f(b)|."""

    scores = np.asarray(option_scores, dtype=np.float64)
    posterior = stable_softmax(posterior_logits)
    if scores.ndim != 2 or scores.shape[1] != len(posterior):
        raise ValueError("option scores must align with the candidate posterior")
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("option scores must be non-empty and finite")
    total = 0.0
    for left in range(len(posterior)):
        for right in range(left + 1, len(posterior)):
            total += float(
                posterior[left]
                * posterior[right]
                * np.mean(np.abs(scores[:, left] - scores[:, right]))
            )
    return total


def ranked_indices(
    values: Sequence[float], count: int, *, exclude: int | None = None
) -> list[int]:
    """Descending score order with earlier-action deterministic ties."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("ranking scores must be a finite vector")
    candidates = [index for index in range(len(scores)) if index != exclude]
    if count <= 0 or len(candidates) < count:
        raise ValueError("requested rank count is outside the candidate inventory")
    return sorted(candidates, key=lambda index: (-float(scores[index]), index))[:count]


def build_nested_views(
    candidate_frame_indices: Sequence[int],
    frame_embeddings: np.ndarray,
    option_text_embeddings: np.ndarray,
    core_frame_indices: Sequence[int],
    core_option_logits: Sequence[float],
    *,
    generic_text_embedding: np.ndarray | None = None,
) -> NestedViews:
    """Build E32/E40/E48 from the immutable core and its own posterior.

    The minimum-valued action is held out as the frozen matched control.  The
    remaining actions are ranked once; E40 takes the first two and E48 takes
    the first four.  If fewer than 32 residual frames remain, both expansions
    stop at the parent core.
    """

    candidates = [int(value) for value in candidate_frame_indices]
    core = sorted({int(value) for value in core_frame_indices})
    if not set(core).issubset(candidates):
        raise ValueError("core frames must belong to the candidate catalog")
    embeddings = np.asarray(frame_embeddings, dtype=np.float32)
    options = np.asarray(option_text_embeddings, dtype=np.float32)
    if len(embeddings) != len(candidates):
        raise ValueError("frame embeddings must align with candidate frames")
    if options.ndim != 2 or len(core_option_logits) != options.shape[0]:
        raise ValueError("option embeddings and posterior logits must align")

    residual = residual_positions(candidates, core)
    if len(residual) < 32:
        return NestedViews(
            tuple(core),
            tuple(core),
            tuple(core),
            tuple(),
            tuple(),
            None,
            tuple(),
            "stop_at_parent",
        )

    windows = action_windows(residual, window_count=8, frames_per_window=4)
    if generic_text_embedding is None:
        # Only candidate-conditioned scores are used by Module II.  Supply a
        # harmless placeholder to reuse the audited cosine-scoring function.
        generic_text_embedding = np.zeros(embeddings.shape[1], dtype=np.float32)
    _, option_scores = retrieval_scores(
        embeddings, np.asarray(generic_text_embedding), options
    )
    values = [
        posterior_pair_discrimination(option_scores[window], core_option_logits)
        for window in windows
    ]
    # Frozen tie rule: if minima tie, hold out the earliest action.
    control = min(range(8), key=lambda index: (values[index], index))
    ranked = ranked_indices(values, 4, exclude=control)

    mapped_actions = [tuple(candidates[position] for position in window) for window in windows]
    e40_added = {frame for action in ranked[:2] for frame in mapped_actions[action]}
    e48_added = {frame for action in ranked[:4] for frame in mapped_actions[action]}
    e40 = tuple(sorted(set(core) | e40_added))
    e48 = tuple(sorted(set(core) | e48_added))
    if len(e40) != len(core) + 8 or len(e48) != len(core) + 16:
        raise RuntimeError("nested refinement violated its exact frame budget")
    if not set(core).issubset(e40) or not set(e40).issubset(e48):
        raise AssertionError("expected E32 subset E40 subset E48")
    return NestedViews(
        tuple(core),
        e40,
        e48,
        tuple(mapped_actions),
        tuple(float(value) for value in values),
        control,
        tuple(ranked),
        None,
    )

