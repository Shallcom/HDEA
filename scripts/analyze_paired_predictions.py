#!/usr/bin/env python3
"""Recompute paired metrics and physical-video-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hdea.io import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("videomme", "longvideobench", "mlvu"), required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evaluated", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20270827)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def indexed(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        qid = str(row["question_id"])
        if qid in rows:
            raise ValueError(f"duplicate question_id in {path}: {qid}")
        rows[qid] = row
    return rows


def prediction_index(row: dict[str, Any]) -> int:
    prediction = row["prediction"]
    if isinstance(prediction, int):
        return prediction
    labels = list(row.get("option_labels", []))
    if labels:
        return labels.index(str(prediction))
    return ord(str(prediction).strip().upper()) - ord("A")


def exact_mcnemar(fixes: int, losses: int) -> float:
    discordant = fixes + losses
    if discordant == 0:
        return 1.0
    tail = min(fixes, losses)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def metric(dataset: str, rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError("metric received no questions")
    if dataset != "mlvu":
        return 100.0 * float(np.mean([row[key] for row in rows]))
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_family"])].append(bool(row[key]))
    if len(grouped) != 7:
        raise ValueError(f"MLVU requires seven task families, found {len(grouped)}")
    return 100.0 * float(np.mean([np.mean(values) for values in grouped.values()]))


def sampled_rows(
    dataset: str,
    rows_by_video: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if dataset != "videomme":
        videos = sorted(rows_by_video)
        sampled = rng.choice(videos, size=len(videos), replace=True)
        return [row for video in sampled for row in rows_by_video[str(video)]]

    by_duration: dict[str, list[str]] = defaultdict(list)
    for video, rows in rows_by_video.items():
        durations = {str(row["duration_group"]) for row in rows}
        if len(durations) != 1:
            raise ValueError(f"VideoMME video crosses duration groups: {video}")
        by_duration[next(iter(durations))].append(video)
    sampled_rows_: list[dict[str, Any]] = []
    for duration in sorted(by_duration):
        videos = sorted(by_duration[duration])
        sampled = rng.choice(videos, size=len(videos), replace=True)
        sampled_rows_.extend(
            row for video in sampled for row in rows_by_video[str(video)]
        )
    return sampled_rows_


def main() -> None:
    args = parse_args()
    labels = indexed(args.labels)
    baseline = indexed(args.baseline)
    evaluated = indexed(args.evaluated)
    if set(labels) != set(baseline) or set(labels) != set(evaluated):
        raise ValueError(
            "inventory mismatch: labels, baseline, and evaluated question sets must be identical"
        )

    rows = []
    for qid in sorted(labels):
        label = labels[qid]
        gold = int(label["gold_index"])
        rows.append(
            {
                "question_id": qid,
                "video_id": str(label["video_id"]),
                "task_family": label.get("task_family"),
                "duration_group": label.get("duration_group"),
                "baseline_correct": prediction_index(baseline[qid]) == gold,
                "evaluated_correct": prediction_index(evaluated[qid]) == gold,
            }
        )

    base_metric = metric(args.dataset, rows, "baseline_correct")
    eval_metric = metric(args.dataset, rows, "evaluated_correct")
    fixes = sum((not row["baseline_correct"]) and row["evaluated_correct"] for row in rows)
    losses = sum(row["baseline_correct"] and (not row["evaluated_correct"]) for row in rows)
    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_video[row["video_id"]].append(row)

    rng = np.random.Generator(np.random.PCG64(args.seed))
    deltas = np.empty(args.replicates, dtype=np.float64)
    for replicate in range(args.replicates):
        sample = sampled_rows(args.dataset, rows_by_video, rng)
        deltas[replicate] = metric(
            args.dataset, sample, "evaluated_correct"
        ) - metric(args.dataset, sample, "baseline_correct")

    result = {
        "dataset": args.dataset,
        "questions": len(rows),
        "unique_videos": len(rows_by_video),
        "baseline_metric": base_metric,
        "evaluated_metric": eval_metric,
        "delta_percentage_points": eval_metric - base_metric,
        "fixes": fixes,
        "losses": losses,
        "net_fixes": fixes - losses,
        "exact_two_sided_mcnemar_p": exact_mcnemar(fixes, losses),
        "bootstrap": {
            "cluster": "physical_video",
            "replicates": args.replicates,
            "rng": "NumPy Generator(PCG64)",
            "seed": args.seed,
            "standard_error": float(deltas.std(ddof=1)),
            "ci_95_percentile": [
                float(np.quantile(deltas, 0.025)),
                float(np.quantile(deltas, 0.975)),
            ],
            "p_delta_gt_0": float(np.mean(deltas > 0)),
        },
        "multiple_comparison_correction": "none",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

