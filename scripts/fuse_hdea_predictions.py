#!/usr/bin/env python3
"""Apply the frozen equal-weight geometric HDEA readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hdea.fusion import geometric_pool
from hdea.io import iter_jsonl


def one_row(path: Path) -> dict:
    rows = list(iter_jsonl(path))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one prediction in {path}")
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--e40", type=Path, required=True)
    parser.add_argument("--e48", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components = [one_row(path) for path in (args.core, args.e40, args.e48)]
    qids = {str(row["question_id"]) for row in components}
    if len(qids) != 1:
        raise ValueError("component predictions refer to different questions")
    log_probs, probabilities, predicted = geometric_pool(
        [row["option_log_probs"] for row in components]
    )
    labels = list(components[0].get("option_labels", []))
    if len(labels) != len(probabilities):
        labels = [chr(ord("A") + index) for index in range(len(probabilities))]
    output = {
        "dataset": components[0]["dataset"],
        "video_id": components[0]["video_id"],
        "question_id": components[0]["question_id"],
        "selector": "full_exact",
        "component_arms": ["hdea_core_32", "e40_exact", "e48_exact"],
        "option_labels": labels,
        "option_log_probs": log_probs.tolist(),
        "option_probs": probabilities.tolist(),
        "prediction": labels[predicted],
        "answer_calls": 3,
        "aggregation_rule": (
            "equal_weight_geometric_pool_of_core_e40_e48_candidate_probabilities"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

