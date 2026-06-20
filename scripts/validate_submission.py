#!/usr/bin/env python3
"""Validate hackathon submission CSV format."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REQUIRED_COLUMNS = ("candidate_id", "rank", "score", "reasoning")
EXPECTED_ROWS = 100


def validate_submission(path: Path) -> list[str]:
    errors: list[str] = []

    if not path.exists():
        return [f"File not found: {path}"]

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return ["CSV has no header row."]

        header = [h.strip() for h in reader.fieldnames]
        if tuple(header) != REQUIRED_COLUMNS:
            errors.append(
                f"Header must be exactly: {','.join(REQUIRED_COLUMNS)}; got {header}"
            )

        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        errors.append(f"Expected exactly {EXPECTED_ROWS} rows, got {len(rows)}")

    ranks: list[int] = []
    scores: list[float] = []
    ids: list[str] = []

    for i, row in enumerate(rows, start=2):
        cid = (row.get("candidate_id") or "").strip()
        if not cid:
            errors.append(f"Line {i}: missing candidate_id")

        try:
            rank = int(row.get("rank", ""))
        except ValueError:
            errors.append(f"Line {i}: invalid rank '{row.get('rank')}'")
            continue
        ranks.append(rank)

        try:
            score = float(row.get("score", ""))
        except ValueError:
            errors.append(f"Line {i}: invalid score '{row.get('score')}'")
            continue
        if score < 0 or score > 100:
            errors.append(f"Line {i}: score {score} out of range 0-100")
        scores.append(score)

        reasoning = (row.get("reasoning") or "").strip()
        if len(reasoning) < 20:
            errors.append(f"Line {i}: reasoning too short or empty")

        ids.append(cid)

    expected_ranks = list(range(1, EXPECTED_ROWS + 1))
    if sorted(ranks) != expected_ranks:
        errors.append(f"Ranks must be 1..{EXPECTED_ROWS} each exactly once.")

    if len(set(ids)) != len(ids):
        errors.append("Duplicate candidate_id values found.")

    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1] + 1e-9:
            errors.append(
                f"Scores must be non-increasing by rank; "
                f"rank {i} score {scores[i-1]:.2f} < rank {i+1} score {scores[i]:.2f}"
            )

    # Tie-break rule: if scores equal, candidate_id should be ascending
    for i in range(1, len(rows)):
        if abs(scores[i] - scores[i - 1]) < 1e-9 and ids[i] < ids[i - 1]:
            errors.append(
                f"Tie-break violation at ranks {i} and {i+1}: "
                f"higher rank must have lexicographically >= candidate_id on ties."
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ranked_output.csv submission.")
    parser.add_argument("csv", type=Path, nargs="?", default=Path("submission/ranked_output.csv"))
    args = parser.parse_args()

    errors = validate_submission(args.csv)
    if errors:
        print("VALIDATION FAILED")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"VALIDATION PASSED: {args.csv}")


if __name__ == "__main__":
    main()
