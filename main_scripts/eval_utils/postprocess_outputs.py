#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _coerce_tsv_value(value: str) -> Any:
    if value == "":
        return None
    try:
        as_int = int(value)
    except ValueError:
        pass
    else:
        return as_int
    try:
        return float(value)
    except ValueError:
        return value


def rewrite_jsonl(path: Path) -> None:
    if not path.exists():
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    changed = False
    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                dst.write(line)
                continue
            payload = json.loads(line)
            new_line = json.dumps(payload, ensure_ascii=False) + "\n"
            changed = changed or new_line != line
            dst.write(new_line)
    if changed:
        tmp_path.replace(path)
    else:
        tmp_path.unlink(missing_ok=True)


def write_scores_json(scores_tsv: Path, scores_json: Path | None = None) -> None:
    if not scores_tsv.exists():
        return
    if scores_json is None:
        scores_json = scores_tsv.with_name("scores.json")
    with scores_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        row = next(reader, None)
    payload = {} if row is None else {key: _coerce_tsv_value(value) for key, value in row.items()}
    scores_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_json_suffix(line: str) -> str:
    stripped = line.rstrip("\n")
    newline = "\n" if line.endswith("\n") else ""
    candidates = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if not candidates:
        return line
    start = min(candidates)
    prefix = stripped[:start]
    suffix = stripped[start:]
    try:
        payload = json.loads(suffix)
    except json.JSONDecodeError:
        return line
    return prefix + json.dumps(payload, ensure_ascii=False, indent=None) + newline


def rewrite_json_prefixed_log(path: Path) -> None:
    if not path.exists():
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    changed = False
    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            new_line = _rewrite_json_suffix(line)
            changed = changed or new_line != line
            dst.write(new_line)
    if changed:
        tmp_path.replace(path)
    else:
        tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process eval outputs as readable UTF-8 JSON.")
    parser.add_argument(
        "--jsonl",
        action="append",
        type=Path,
        default=[],
        help="JSONL file to rewrite with ensure_ascii=False",
    )
    parser.add_argument("--scores-tsv", type=Path, help="SimulEval scores.tsv to mirror as scores.json")
    parser.add_argument("--scores-json", type=Path, help="Optional scores.json output path")
    parser.add_argument(
        "--json-prefixed-log",
        action="append",
        type=Path,
        default=[],
        help="Log file whose lines may contain a JSON object/array suffix to rewrite as UTF-8",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.jsonl:
        rewrite_jsonl(path)
    if args.scores_tsv is not None:
        write_scores_json(args.scores_tsv, args.scores_json)
    for path in args.json_prefixed_log:
        rewrite_json_prefixed_log(path)


if __name__ == "__main__":
    main()
