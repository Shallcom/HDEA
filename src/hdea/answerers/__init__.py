"""Restricted-option scoring adapters used in the frozen experiments.

Heavy model dependencies are imported lazily by each scorer.  Model locations
are supplied through the ``HDEA_*_MODEL`` environment variables documented in
``configs/models.example.yaml``.
"""

from .qwen3 import Qwen3MCQScorer, build_mcq_prompt

__all__ = ["Qwen3MCQScorer", "build_mcq_prompt"]
