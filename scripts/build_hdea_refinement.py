#!/usr/bin/env python3
"""Build exact E40/E48 rows from a frozen HDEA core and its own logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hdea.io import iter_jsonl, selection_hash, write_jsonl
from hdea.refinement import build_nested_views


def one_row(path: Path) -> dict:
    rows = list(iter_jsonl(path))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--core-selection", type=Path, required=True)
    parser.add_argument("--core-prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core = one_row(args.core_selection)
    prediction = one_row(args.core_prediction)
    if str(core["question_id"]) != str(prediction["question_id"]):
        raise ValueError("core selection/prediction question mismatch")
    with np.load(args.cache) as cache:
        frame_indices = [int(value) for value in cache["frame_indices"].tolist()]
        frame_embeddings = cache["frame_embeddings"].astype(np.float32)
        option_embeddings = cache["option_text_embeddings"].astype(np.float32)
        generic_embedding = cache["generic_text_embedding"].astype(np.float32)
        fps = float(cache["fps"].reshape(-1)[0])

    views = build_nested_views(
        frame_indices,
        frame_embeddings,
        option_embeddings,
        core["selected_frame_ids"],
        prediction["option_logits"],
        generic_text_embedding=generic_embedding,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for selector, frames, selected_count in (
        ("e40_exact", views.e40, 2),
        ("e48_exact", views.e48, 4),
    ):
        timestamps = [float(frame / fps) for frame in frames]
        selected_actions = list(views.ranked_actions[:selected_count])
        added = sorted(
            {
                frame
                for action in selected_actions
                for frame in views.residual_actions[action]
            }
        )
        row = {
            "dataset": str(core["dataset"]),
            "video_id": str(core["video_id"]),
            "question_id": str(core["question_id"]),
            "selector": selector,
            "parent_selector": "hdea_core_32",
            "parent_core_selection_hash": str(core["selection_hash"]),
            "selected_action_indices": selected_actions,
            "action_values": list(views.action_values),
            "control_action_index": views.control_action,
            "added_frame_ids": added,
            "selected_frame_ids": list(frames),
            "selected_timestamps": timestamps,
            "actual_frame_count": len(frames),
            "fallback": views.fallback,
        }
        row["selection_hash"] = selection_hash(
            row["dataset"], row["question_id"], frames, timestamps
        )
        write_jsonl(args.output_dir / f"{selector}.jsonl", [row])
        print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

