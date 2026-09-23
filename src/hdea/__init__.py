"""HDEA: hypothesis-discriminative evidence acquisition for long-video QA."""

from .core import (
    CoreSelection,
    adaptive_temporal_leaves,
    explicit_video_timestamps,
    fixed_local_packets,
    pairwise_discrimination,
    select_hdea_core,
)
from .fusion import geometric_pool
from .refinement import NestedViews, build_nested_views

__all__ = [
    "CoreSelection",
    "NestedViews",
    "adaptive_temporal_leaves",
    "build_nested_views",
    "explicit_video_timestamps",
    "fixed_local_packets",
    "geometric_pool",
    "pairwise_discrimination",
    "select_hdea_core",
]

