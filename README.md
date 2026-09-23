# HDEA

Official code release for **HDEA: Hypothesis-Discriminative Evidence
Acquisition for Long-Video Question Answering**.

HDEA is a training-free, answer-model-agnostic method that constructs visual
context for frozen multiple-choice Video-MLLMs. It has two modules:

1. **Hypothesis-Discriminative Core Acquisition** selects a 32-frame evidence
   portfolio by covering complementary candidate-answer distinctions rather
   than independently ranking frames.
2. **Core-Preserving Nested Refinement** uses the core answer posterior to
   rank fixed residual evidence groups, constructs nested E40/E48 views, and
   applies a fixed geometric answer readout.

The repository contains the frozen method implementation, restricted-option
adapters for four Video-MLLMs, deterministic manifest scripts, tests, frozen
configs, aggregate results, and final audit summaries. It intentionally does
not contain model weights, benchmark videos, private paths, large feature
caches, or historical failed experiments.

## Quick start

```bash
pip install -e '.[dev]'
pytest -q
```

```python
import numpy as np
from hdea.core import retrieval_scores, select_hdea_core

generic_scores, option_scores = retrieval_scores(
    frame_embeddings,
    generic_question_embedding,
    candidate_conditioned_embeddings,
)
core = select_hdea_core(generic_scores, option_scores, frame_budget=32)
selected_physical_frames = [candidate_frame_ids[i] for i in core.frame_positions]
```

For the exact two-stage sequence, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md)
and [docs/METHOD.md](docs/METHOD.md).

## Frozen protocol

- Frozen retriever: SigLIP-So400M patch14-384
- Candidate catalog: deterministic integer 1 FPS
- Local evidence unit: adjacent non-overlapping two-frame packet
- Core: 16 packets / at most 32 source frames
- Nested views: E32 -> E40 -> E48
- Final readout: equal-weight geometric pool of the three option distributions
- No training, gold label, correctness, ASR, subtitle, captioner, or external
  generative model is used by the selector

The full protocol and selection-set hashes are in
[`configs/hdea_frozen.yaml`](configs/hdea_frozen.yaml).

## Main frozen results

| Backbone | VideoMME | LongVideoBench | MLVU |
|---|---:|---:|---:|
| Qwen3-VL-8B | **67.44** | **65.22** | **74.25** |
| InternVL3-8B | **66.33** | **64.47** | **73.50** |
| mPLUG-Owl3-7B | **55.96** | **58.34** | **65.22** |
| LLaVA-NeXT-Video-7B | **62.48** | **60.88** | **68.65** |

VideoMME and LongVideoBench report accuracy. MLVU reports its official
unweighted seven-task M-AVG. See [docs/RESULTS.md](docs/RESULTS.md) and the CSV
files under `results/` for matched baselines, ablations, paired tests, and
physical-video-cluster bootstrap intervals.

## Repository layout

```text
src/hdea/                 Frozen Module-I/II algorithms and answerer adapters
scripts/                  Portable manifest, refinement, fusion, and audit CLIs
configs/                  Frozen method protocol and model-path template
tests/                    Determinism, invariance, nesting, and regression tests
results/module1/          Selector decomposition and cross-model Core tables
results/module2/          Exact-common-parent Module-II tables
results/audits/           Final integrity audit summaries
docs/                     Method, reproduction, and result documentation
```

## Data and checkpoints

Benchmark videos and model checkpoints are not redistributed. Obtain them
from their official providers and set the model paths shown in
[`configs/models.example.yaml`](configs/models.example.yaml). The public code
operates on an explicit frozen embedding-cache schema, so selection can be
audited without exposing labels to the selector.

## Citation

Author-identifying citation metadata is intentionally omitted during
double-blind review. The BibTeX entry and archival release identifier will be
added after the paper record is public.

## Licence

A software licence must be selected by the repository owner before the first
public GitHub release. No licence is inferred automatically.
