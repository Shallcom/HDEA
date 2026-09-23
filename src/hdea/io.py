"""Outcome-free manifest and hashing helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


FORBIDDEN_SELECTION_KEYS = {
    "gold",
    "gold_answer",
    "correct",
    "correctness",
    "baseline_correct",
    "target_label",
    "task_accuracy",
}


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_hash(
    dataset: str,
    question_id: str,
    frame_ids: Iterable[int],
    timestamps: Iterable[float],
) -> str:
    return sha256_json(
        {
            "dataset": dataset,
            "question_id": str(question_id),
            "selected_frame_ids": [int(value) for value in frame_ids],
            "selected_timestamps": [round(float(value), 7) for value in timestamps],
        }
    )


def forbidden_paths(value: Any, path: str = "root") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_SELECTION_KEYS:
                violations.append(child)
            violations.extend(forbidden_paths(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            violations.extend(forbidden_paths(nested, f"{path}[{index}]"))
    return violations


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            violations = forbidden_paths(row)
            if violations:
                raise ValueError(f"outcome-derived fields in selection: {violations[:3]}")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)

