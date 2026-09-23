#!/usr/bin/env python3
"""Build one outcome-free HDEA-Core manifest row from frozen embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hdea.core import (
    retrieval_scores,
    select_hdea_core,
    select_required_positions,
)
from hdea.io import selection_hash, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-budget", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    with np.load(args.cache) as cache:
        frame_indices = [int(value) for value in cache["frame_indices"].tolist()]
        frame_embeddings = cache["frame_embeddings"].astype(np.float32)
        generic_embedding = cache["generic_text_embedding"].astype(np.float32)
        option_embeddings = cache["option_text_embeddings"].astype(np.float32)
        fps = float(cache["fps"].reshape(-1)[0])
        total_frames = int(cache["total_frames"].reshape(-1)[0])

    generic, options = retrieval_scores(
        frame_embeddings, generic_embedding, option_embeddings
    )
    required = select_required_positions(
        metadata["question"], frame_indices, fps, total_frames
    )
    core = select_hdea_core(
        generic,
        options,
        args.frame_budget,
        required_frame_positions=required,
    )
    frames = [frame_indices[position] for position in core.frame_positions]
    timestamps = [float(frame / fps) for frame in frames]
    row = {
        "dataset": str(metadata["dataset"]),
        "video_id": str(metadata["video_id"]),
        "question_id": str(metadata["question_id"]),
        "question": str(metadata["question"]),
        "candidates": [str(value) for value in metadata["candidates"]],
        "selector": "hdea_core_32",
        "selected_packet_ids": [f"p{value:06d}" for value in core.packet_indices],
        "required_packet_ids": [
            f"p{value:06d}" for value in core.required_packet_indices
        ],
        "selected_frame_positions": list(core.frame_positions),
        "selected_frame_ids": frames,
        "selected_timestamps": timestamps,
        "actual_frame_count": len(frames),
        "candidate_frame_count": len(frame_indices),
        "hdea_set_objective": core.facility_coverage,
    }
    row["selection_hash"] = selection_hash(
        row["dataset"], row["question_id"], frames, timestamps
    )
    write_jsonl(args.output, [row])
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

