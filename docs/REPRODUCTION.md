# Reproduction guide

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy torch transformers pillow pandas scipy pyarrow pytest
PYTHONPATH=src pytest -q
```

## 2. Prepare one retrieval cache

Decode the video on a deterministic integer one-frame-per-second catalog and
save one compressed NumPy archive with these arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `frame_indices` | `[N]` | Physical decoded frame IDs |
| `frame_embeddings` | `[N,D]` | Frozen SigLIP image embeddings |
| `generic_text_embedding` | `[D]` | `Question: {question}` |
| `option_text_embeddings` | `[M,D]` | One embedding per answer option |
| `fps` | `[1]` | Decoded video FPS |
| `total_frames` | `[1]` | Decoded physical frame count |

Text templates and the retriever wrapper are in `hdea.retrieval`. Metadata is
a JSON object containing `dataset`, `video_id`, `question_id`, `question`, and
`candidates`. No answer label belongs in either selection input.

## 3. Build HDEA-Core

```bash
python scripts/build_hdea_core.py \
  --cache example.npz \
  --metadata example.json \
  --output core.jsonl
```

Score the selected frames once with a restricted-option answerer and save its
raw candidate logits and log probabilities. Model adapters matching the
frozen interfaces are under `hdea.answerers`.

## 4. Build the nested views

```bash
python scripts/build_hdea_refinement.py \
  --cache example.npz \
  --core-selection core.jsonl \
  --core-prediction core_prediction.jsonl \
  --output-dir nested/
```

Score `nested/e40_exact.jsonl` and `nested/e48_exact.jsonl` with the same
answer model and prompt as the core.

## 5. Fuse predictions

```bash
python scripts/fuse_hdea_predictions.py \
  --core core_prediction.jsonl \
  --e40 e40_prediction.jsonl \
  --e48 e48_prediction.jsonl \
  --output full_prediction.jsonl
```

## Benchmark data

This repository does not redistribute benchmark videos, annotations, model
weights, or large feature caches. Download VideoMME, LongVideoBench, and MLVU
from their official sources and comply with their licences. Aggregate frozen
statistics are provided under `results/`; selection/prediction artifacts can
be released separately if their dataset terms permit redistribution.
