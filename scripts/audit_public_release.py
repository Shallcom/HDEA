#!/usr/bin/env python3
"""Fail closed on common accidental-publication problems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


FORBIDDEN_SUFFIXES = {
    ".avi",
    ".mkv",
    ".mp4",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
ANONYMOUS_EMAIL = "anonymous" + "@invalid.example"
PATTERNS = {
    "absolute_workspace_path": re.compile(
        "/ho" + "me/" + "workspace/" + "|/ro" + "ot/" + "anaconda"
    ),
    "personal_home_path": re.compile(
        r"(?:/ho"
        + r"me/(?!workspace(?:/|$))[^/\s]+|/Us"
        + r"ers/[^/\s]+|[A-Za-z]:\\Us"
        + r"ers\\[^\\\s]+)"
    ),
    "email_address": re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ),
    "experiment_tracking_identity": re.compile(
        r"(?i)\b("
        + "wan"
        + r"db|weights[ ._-]*and[ ._-]*biases|co"
        + r"met_ml|nep"
        + r"tune\.ai)\b"
    ),
    "credential_like_assignment": re.compile(
        r"(?i)\b(token|password|secret|api[_-]?key)\b\s*[=:]\s*['\"][^'\"]+"
    ),
}

ANONYMOUS_GIT_IDENTITY = (
    "Anonymous Authors",
    ANONYMOUS_EMAIL,
    "Anonymous Authors",
    ANONYMOUS_EMAIL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    root = parse_args().root.resolve()
    violations: list[dict[str, str]] = []
    files = [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]
    for path in files:
        relative = path.relative_to(root)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append({"file": str(relative), "reason": "forbidden binary artifact"})
        if path.stat().st_size > 25 * 1024 * 1024:
            violations.append({"file": str(relative), "reason": "file exceeds 25 MiB"})
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            text = text.replace(ANONYMOUS_EMAIL, "")
            for name, pattern in PATTERNS.items():
                if pattern.search(text):
                    violations.append({"file": str(relative), "reason": name})

    if (root / ".git").exists():
        history = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "--all",
                "--format=%an%x09%ae%x09%cn%x09%ce",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line_number, line in enumerate(history.stdout.splitlines(), start=1):
            identity = tuple(line.split("\t"))
            if identity != ANONYMOUS_GIT_IDENTITY:
                violations.append(
                    {
                        "file": ".git",
                        "reason": f"non-anonymous Git identity at history row {line_number}",
                    }
                )
    result = {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "passed": not violations,
        "violations": violations,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
