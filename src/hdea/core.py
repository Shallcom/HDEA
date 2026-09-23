"""Frozen Module-I implementation used in the HDEA experiments.

The selector is outcome free.  It receives generic and candidate-conditioned
retrieval scores, constructs non-overlapping two-frame evidence units, and
greedily maximizes candidate-pair facility coverage inside the frozen adaptive
temporal partition.  It never receives a gold answer or model correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import re
from typing import Iterable, Sequence

import numpy as np


_CLOCK_PATTERN = re.compile(
    r"(?<!\d)(?:(?P<hour>\d{1,2}):)?(?P<minute>[0-5]?\d):(?P<second>[0-5]\d)(?!\d)"
)
_CLOCK_PREFIX = re.compile(
    r"\b(?:at|from|between|around|during)"
    r"(?:\s+(?:about|approximately|exactly))?\s*$",
    flags=re.IGNORECASE,
)
_CLOCK_CONTINUATION = re.compile(r"^\s*(?:-|–|—|to|and)\s*$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class TemporalLeaf:
    start: int
    stop: int
    depth: int
    quota: int


@dataclass(frozen=True)
class CoreSelection:
    """Auditable Module-I output in candidate-catalog coordinates."""

    frame_positions: tuple[int, ...]
    packet_indices: tuple[int, ...]
    required_packet_indices: tuple[int, ...]
    facility_coverage: float


def explicit_video_timestamps(question: str) -> list[int]:
    """Parse only query-time clock constraints, never evidence annotations."""

    matches = list(_CLOCK_PATTERN.finditer(str(question)))
    accepted: list[int] = []
    previous_match: re.Match[str] | None = None
    previous_accepted = False
    for match in matches:
        prefix = str(question)[max(0, match.start() - 32) : match.start()]
        continuation = (
            previous_match is not None
            and previous_accepted
            and bool(
                _CLOCK_CONTINUATION.fullmatch(
                    str(question)[previous_match.end() : match.start()]
                )
            )
        )
        use = bool(_CLOCK_PREFIX.search(prefix)) or continuation
        if use:
            hour = int(match.group("hour") or 0)
            minute = int(match.group("minute"))
            second = int(match.group("second"))
            accepted.append(hour * 3600 + minute * 60 + second)
        previous_match = match
        previous_accepted = use
    return list(dict.fromkeys(accepted))


def minmax_normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if array.size == 0:
        return array
    low = float(np.min(array))
    high = float(np.max(array))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("scores must be finite")
    if high == low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def adaptive_temporal_leaves(
    scores: Sequence[float],
    budget: int,
    *,
    threshold: float = 0.2,
    max_depth: int = 5,
) -> list[TemporalLeaf]:
    """Create the frozen adaptive temporal leaves and inherited quotas."""

    normalized = minmax_normalize(scores)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    if len(normalized) == 0:
        return []
    leaves: list[TemporalLeaf] = []

    def recurse(start: int, stop: int, depth: int) -> None:
        segment = normalized[start:stop]
        nominal_quota = max(1, budget // (2**depth))
        quota = min(nominal_quota, len(segment))
        if len(segment) <= 1:
            leaves.append(TemporalLeaf(start, stop, depth, quota))
            return
        top_count = min(budget, len(segment))
        top = heapq.nlargest(top_count, range(len(segment)), segment.__getitem__)
        mean_difference = float(np.mean(segment[top]) - np.mean(segment))
        if mean_difference > threshold or depth >= max_depth:
            leaves.append(TemporalLeaf(start, stop, depth, quota))
            return
        middle = start + len(segment) // 2
        recurse(start, middle, depth + 1)
        recurse(middle, stop, depth + 1)

    recurse(0, len(normalized), 0)
    return leaves


def fixed_local_packets(length: int) -> list[tuple[int, int]]:
    """Return the frozen packet bank: (0,1), (2,3), ... without frame reuse."""

    return [(index, index + 1) for index in range(0, length - 1, 2)]


def pairwise_discrimination(option_scores: np.ndarray) -> np.ndarray:
    """Return |z_i(c_a)-z_i(c_b)| for every unit and unordered option pair."""

    values = np.asarray(option_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("option_scores must have shape [units, choices>=2]")
    columns = []
    for first in range(values.shape[1]):
        for second in range(first + 1, values.shape[1]):
            columns.append(np.abs(values[:, first] - values[:, second]))
    return np.stack(columns, axis=1)


def facility_coverage(facilities: np.ndarray, positions: Sequence[int]) -> float:
    values = np.asarray(facilities, dtype=np.float64)
    chosen = [int(value) for value in positions]
    if not chosen:
        return 0.0
    if min(chosen) < 0 or max(chosen) >= len(values):
        raise IndexError("coverage position is outside the facility catalog")
    return float(values[chosen].max(axis=0).sum())


def _facility_greedy(
    generic_scores: np.ndarray,
    facilities: np.ndarray,
    leaves: Sequence[TemporalLeaf],
    budget: int,
    *,
    required_units: Sequence[int] = (),
) -> list[int]:
    """Frozen set-dependent greedy rule with seven-decimal deterministic ties."""

    generic = np.asarray(generic_scores, dtype=np.float64)
    discrimination = np.asarray(facilities, dtype=np.float64)
    if discrimination.ndim != 2 or discrimination.shape[1] == 0:
        raise ValueError("facilities must have shape [units, facilities>=1]")
    if len(discrimination) != len(generic):
        raise ValueError("generic scores and facility rows differ in length")
    if np.any(~np.isfinite(discrimination)) or np.any(discrimination < 0):
        raise ValueError("coverage facilities must be finite and nonnegative")
    required = sorted({int(value) for value in required_units})
    if len(required) > budget:
        raise ValueError("required units exceed the acquisition budget")
    if required and (required[0] < 0 or required[-1] >= len(generic)):
        raise IndexError("required acquisition unit is out of range")

    current = (
        discrimination[required].max(axis=0)
        if required
        else np.zeros(discrimination.shape[1], dtype=np.float64)
    )
    selected = list(required)
    selected_set = set(required)
    remaining = {index: leaf.quota for index, leaf in enumerate(leaves)}
    unit_leaf: dict[int, int] = {}
    for leaf_index, leaf in enumerate(leaves):
        for unit in range(leaf.start, leaf.stop):
            unit_leaf[unit] = leaf_index
    for unit in required:
        leaf_index = unit_leaf[unit]
        remaining[leaf_index] = max(0, remaining[leaf_index] - 1)

    while len(selected) < budget:
        best_unit = None
        best_key = None
        for unit in range(len(generic)):
            if unit in selected_set:
                continue
            leaf_index = unit_leaf[unit]
            if remaining[leaf_index] <= 0:
                continue
            marginal = float(
                np.maximum(current, discrimination[unit]).sum() - current.sum()
            )
            key = (
                round(marginal, 7),
                round(float(generic[unit]), 7),
                -unit,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_unit = unit
        if best_unit is None:
            break
        selected.append(best_unit)
        selected_set.add(best_unit)
        remaining[unit_leaf[best_unit]] -= 1
        current = np.maximum(current, discrimination[best_unit])

    if len(selected) != budget:
        raise RuntimeError(
            f"HDEA selection could fill only {len(selected)} of {budget} units"
        )
    return sorted(selected)


def _modular_greedy(
    generic_scores: np.ndarray,
    primary_scores: np.ndarray,
    leaves: Sequence[TemporalLeaf],
    budget: int,
    *,
    required_units: Sequence[int] = (),
) -> list[int]:
    """Matched pointwise control inside exactly the same temporal feasible set."""

    generic = np.asarray(generic_scores, dtype=np.float64)
    primary = np.asarray(primary_scores, dtype=np.float64)
    if generic.ndim != 1 or primary.ndim != 1 or len(generic) != len(primary):
        raise ValueError("generic and primary scores must be aligned vectors")
    if np.any(~np.isfinite(generic)) or np.any(~np.isfinite(primary)):
        raise ValueError("modular scores must be finite")
    required = sorted({int(value) for value in required_units})
    if len(required) > budget:
        raise ValueError("required units exceed the acquisition budget")

    unit_leaf: dict[int, int] = {}
    remaining = {index: leaf.quota for index, leaf in enumerate(leaves)}
    for leaf_index, leaf in enumerate(leaves):
        for unit in range(leaf.start, leaf.stop):
            unit_leaf[unit] = leaf_index
    selected = list(required)
    selected_set = set(required)
    for unit in required:
        remaining[unit_leaf[unit]] = max(0, remaining[unit_leaf[unit]] - 1)

    while len(selected) < budget:
        best_unit = None
        best_key = None
        for unit in range(len(generic)):
            if unit in selected_set or remaining[unit_leaf[unit]] <= 0:
                continue
            key = (
                round(float(primary[unit]), 7),
                round(float(generic[unit]), 7),
                -unit,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_unit = unit
        if best_unit is None:
            break
        selected.append(best_unit)
        selected_set.add(best_unit)
        remaining[unit_leaf[best_unit]] -= 1
    if len(selected) != budget:
        raise RuntimeError(
            f"matched control could fill only {len(selected)} of {budget} units"
        )
    return sorted(selected)


def _packet_arrays(
    generic_frame_scores: Sequence[float], option_frame_scores: np.ndarray
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    generic = np.asarray(generic_frame_scores, dtype=np.float64)
    options = np.asarray(option_frame_scores, dtype=np.float64)
    if generic.ndim != 1 or options.ndim != 2 or len(generic) != len(options):
        raise ValueError("generic and option frame scores must be aligned")
    packets = fixed_local_packets(len(generic))
    packet_generic = np.asarray([max(generic[a], generic[b]) for a, b in packets])
    packet_options = np.asarray(
        [np.maximum(options[a], options[b]) for a, b in packets]
    )
    return packets, packet_generic, packet_options


def select_hdea_core(
    generic_frame_scores: Sequence[float],
    option_frame_scores: np.ndarray,
    frame_budget: int = 32,
    *,
    required_frame_positions: Sequence[int] = (),
    threshold: float = 0.2,
    max_depth: int = 5,
) -> CoreSelection:
    """Construct the HDEA 16-packet/32-frame hypothesis-discriminative core."""

    generic = np.asarray(generic_frame_scores, dtype=np.float64)
    options = np.asarray(option_frame_scores, dtype=np.float64)
    if generic.ndim != 1 or options.ndim != 2 or len(generic) != len(options):
        raise ValueError("generic and option frame scores must be aligned")
    if frame_budget <= 0 or frame_budget % 2:
        raise ValueError("HDEA requires a positive, even frame budget")
    if len(generic) <= frame_budget:
        packets = fixed_local_packets(len(generic))
        packet_options = np.asarray(
            [np.maximum(options[a], options[b]) for a, b in packets]
        )
        facilities = (
            pairwise_discrimination(packet_options)
            if len(packet_options)
            else np.empty((0, options.shape[1] * (options.shape[1] - 1) // 2))
        )
        packet_indices = tuple(range(len(packets)))
        return CoreSelection(
            tuple(range(len(generic))),
            packet_indices,
            tuple(),
            facility_coverage(facilities, packet_indices),
        )

    packets, packet_generic, packet_options = _packet_arrays(generic, options)
    unit_budget = frame_budget // 2
    if len(packets) < unit_budget:
        raise RuntimeError("packet catalog cannot fill the requested budget")
    normalized_generic = minmax_normalize(packet_generic)
    leaves = adaptive_temporal_leaves(
        normalized_generic, unit_budget, threshold=threshold, max_depth=max_depth
    )
    required_packets = sorted(
        {
            min(int(position) // 2, len(packets) - 1)
            for position in required_frame_positions
        }
    )
    facilities = pairwise_discrimination(packet_options)
    chosen = _facility_greedy(
        normalized_generic,
        facilities,
        leaves,
        unit_budget,
        required_units=required_packets,
    )
    positions = tuple(frame for packet_index in chosen for frame in packets[packet_index])
    return CoreSelection(
        frame_positions=positions,
        packet_indices=tuple(chosen),
        required_packet_indices=tuple(required_packets),
        facility_coverage=facility_coverage(facilities, chosen),
    )


def select_pairwise_pointwise(
    generic_frame_scores: Sequence[float],
    option_frame_scores: np.ndarray,
    frame_budget: int = 32,
    *,
    required_frame_positions: Sequence[int] = (),
) -> tuple[int, ...]:
    """Pairwise representation without HDEA's set-level complementarity."""

    generic = np.asarray(generic_frame_scores, dtype=np.float64)
    options = np.asarray(option_frame_scores, dtype=np.float64)
    if len(generic) <= frame_budget:
        return tuple(range(len(generic)))
    packets, packet_generic, packet_options = _packet_arrays(generic, options)
    unit_budget = frame_budget // 2
    normalized_generic = minmax_normalize(packet_generic)
    leaves = adaptive_temporal_leaves(normalized_generic, unit_budget)
    required = sorted(
        {min(int(position) // 2, len(packets) - 1) for position in required_frame_positions}
    )
    pointwise = pairwise_discrimination(packet_options).mean(axis=1)
    chosen = _modular_greedy(
        normalized_generic, pointwise, leaves, unit_budget, required_units=required
    )
    return tuple(frame for packet_index in chosen for frame in packets[packet_index])


def option_pool_text_embedding(option_embeddings: np.ndarray) -> np.ndarray:
    """Normalize(mean_a psi(q,c_a)) with permutation-stable summation."""

    normalized = l2_normalize(option_embeddings)
    order = sorted(range(len(normalized)), key=lambda index: normalized[index].tobytes())
    pooled = normalized[order].astype(np.float64).mean(axis=0, keepdims=True)
    return l2_normalize(pooled)[0]


def select_option_pool(
    generic_frame_scores: Sequence[float],
    pooled_frame_scores: Sequence[float],
    frame_budget: int = 32,
    *,
    required_frame_positions: Sequence[int] = (),
) -> tuple[int, ...]:
    """Candidate-collapsed option-aware matched control."""

    generic = np.asarray(generic_frame_scores, dtype=np.float64)
    pooled = np.asarray(pooled_frame_scores, dtype=np.float64)
    if generic.ndim != 1 or pooled.ndim != 1 or len(generic) != len(pooled):
        raise ValueError("generic and pooled frame scores must be aligned")
    if len(generic) <= frame_budget:
        return tuple(range(len(generic)))
    packets = fixed_local_packets(len(generic))
    packet_generic = np.asarray([max(generic[a], generic[b]) for a, b in packets])
    packet_pool = np.asarray([max(pooled[a], pooled[b]) for a, b in packets])
    unit_budget = frame_budget // 2
    normalized_generic = minmax_normalize(packet_generic)
    leaves = adaptive_temporal_leaves(normalized_generic, unit_budget)
    required = sorted(
        {min(int(position) // 2, len(packets) - 1) for position in required_frame_positions}
    )
    chosen = _modular_greedy(
        normalized_generic, packet_pool, leaves, unit_budget, required_units=required
    )
    return tuple(frame for packet_index in chosen for frame in packets[packet_index])


def l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norm, np.finfo(np.float32).eps)


def retrieval_scores(
    frame_embeddings: np.ndarray,
    generic_text_embedding: np.ndarray,
    option_text_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generic and candidate-conditioned cosine retrieval scores."""

    frames = l2_normalize(frame_embeddings)
    generic = l2_normalize(np.asarray(generic_text_embedding).reshape(1, -1))[0]
    options = l2_normalize(option_text_embeddings)
    if frames.shape[1] != generic.shape[0] or frames.shape[1] != options.shape[1]:
        raise ValueError("retrieval embedding dimensions differ")
    return frames @ generic, frames @ options.T


def select_required_positions(
    question: str,
    candidate_frame_indices: Sequence[int],
    fps: float,
    total_frames: int,
) -> list[int]:
    """Map literal query timestamps to the nearest candidate-catalog positions."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    candidates = np.asarray(candidate_frame_indices, dtype=np.int64)
    if len(candidates) == 0:
        return []
    duration = float(total_frames / fps)
    times = candidates.astype(np.float64) / fps
    explicit = [
        value for value in explicit_video_timestamps(question) if 0 <= value <= duration + 1.0
    ]
    return sorted({int(np.abs(times - value).argmin()) for value in explicit})

