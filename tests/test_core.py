import numpy as np

from hdea.core import (
    fixed_local_packets,
    pairwise_discrimination,
    select_hdea_core,
    select_option_pool,
    select_pairwise_pointwise,
)


def synthetic_scores(seed: int = 7, length: int = 96):
    rng = np.random.default_rng(seed)
    return rng.normal(size=length), rng.normal(size=(length, 4))


def test_core_is_exact_budget_local_and_chronological():
    generic, options = synthetic_scores()
    result = select_hdea_core(generic, options)
    assert len(result.frame_positions) == 32
    assert list(result.frame_positions) == sorted(result.frame_positions)
    assert len(set(result.frame_positions)) == 32
    assert all(
        second == first + 1
        for first, second in zip(result.frame_positions[::2], result.frame_positions[1::2])
    )


def test_core_honors_timestamp_contraction_inside_budget():
    generic, options = synthetic_scores(seed=11)
    result = select_hdea_core(
        generic, options, required_frame_positions=[91]
    )
    assert len(result.frame_positions) == 32
    assert 90 in result.frame_positions and 91 in result.frame_positions
    assert 45 in result.required_packet_indices


def test_candidate_permutation_does_not_change_selection():
    generic, options = synthetic_scores(seed=19)
    original = select_hdea_core(generic, options).frame_positions
    for permutation in ([3, 2, 1, 0], [2, 0, 3, 1]):
        selected = select_hdea_core(generic, options[:, permutation]).frame_positions
        assert selected == original


def test_frozen_regression_selection():
    generic, options = synthetic_scores(seed=23)
    selected = select_hdea_core(generic, options).frame_positions
    assert selected == (
        12, 13, 18, 19, 26, 27, 28, 29, 32, 33, 34, 35, 40, 41, 44, 45,
        48, 49, 56, 57, 64, 65, 66, 67, 80, 81, 84, 85, 86, 87, 88, 89,
    )


def test_matched_controls_are_deterministic():
    generic, options = synthetic_scores(seed=29)
    pointwise = select_pairwise_pointwise(generic, options)
    pooled = select_option_pool(generic, options.max(axis=1))
    assert len(pointwise) == len(pooled) == 32
    assert pointwise == select_pairwise_pointwise(generic, options)
    assert pooled == select_option_pool(generic, options.max(axis=1))


def test_pairwise_shape_and_packet_contract():
    assert fixed_local_packets(7) == [(0, 1), (2, 3), (4, 5)]
    values = pairwise_discrimination(np.ones((5, 4)))
    assert values.shape == (5, 6)
    assert np.all(values == 0)
