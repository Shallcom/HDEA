import numpy as np

from hdea.refinement import action_windows, build_nested_views


def test_action_windows_are_disjoint_and_chronological():
    windows = action_windows(range(40), window_count=8, frames_per_window=4)
    assert len(windows) == 8
    assert all(len(window) == 4 for window in windows)
    flattened = [value for window in windows for value in window]
    assert flattened == sorted(flattened)
    assert len(flattened) == len(set(flattened))


def test_nested_views_preserve_core_and_prefix():
    rng = np.random.default_rng(31)
    candidates = list(range(0, 288, 3))
    embeddings = rng.normal(size=(96, 16)).astype(np.float32)
    option_text = rng.normal(size=(4, 16)).astype(np.float32)
    generic_text = rng.normal(size=16).astype(np.float32)
    core = sorted(rng.choice(candidates, size=32, replace=False).tolist())
    views = build_nested_views(
        candidates,
        embeddings,
        option_text,
        core,
        [1.0, 0.8, -0.2, -0.4],
        generic_text_embedding=generic_text,
    )
    assert len(views.e32) == 32
    assert len(views.e40) == 40
    assert len(views.e48) == 48
    assert set(views.e32).issubset(views.e40)
    assert set(views.e40).issubset(views.e48)
    assert len(views.residual_actions) == 8
    assert views.control_action not in views.ranked_actions


def test_short_residual_catalog_stops_at_parent():
    rng = np.random.default_rng(37)
    candidates = list(range(40))
    embeddings = rng.normal(size=(40, 8)).astype(np.float32)
    option_text = rng.normal(size=(4, 8)).astype(np.float32)
    core = list(range(32))
    views = build_nested_views(
        candidates, embeddings, option_text, core, [0.0, 0.0, 0.0, 0.0]
    )
    assert views.fallback == "stop_at_parent"
    assert views.e32 == views.e40 == views.e48


def test_candidate_permutation_preserves_action_ranking():
    rng = np.random.default_rng(41)
    candidates = list(range(96))
    embeddings = rng.normal(size=(96, 12)).astype(np.float32)
    option_text = rng.normal(size=(4, 12)).astype(np.float32)
    generic_text = rng.normal(size=12).astype(np.float32)
    core = list(range(0, 64, 2))
    logits = np.asarray([2.0, 1.0, 0.0, -1.0])
    original = build_nested_views(
        candidates,
        embeddings,
        option_text,
        core,
        logits,
        generic_text_embedding=generic_text,
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted = build_nested_views(
        candidates,
        embeddings,
        option_text[permutation],
        core,
        logits[permutation],
        generic_text_embedding=generic_text,
    )
    assert original.control_action == permuted.control_action
    assert original.ranked_actions == permuted.ranked_actions
    assert np.allclose(original.action_values, permuted.action_values)

