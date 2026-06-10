import csv
from pathlib import Path

from .io import CSV_COLUMNS


def validate_ranked_output(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}. Expected: {CSV_COLUMNS}")

        rows = list(reader)

    if not rows:
        raise ValueError("Ranked output is empty")

    ranks = [int(row["rank"]) for row in rows]
    expected = list(range(1, len(rows) + 1))
    if ranks != expected:
        raise ValueError(f"Ranks must be consecutive starting at 1. Found {ranks}")

    seen_ids = set()
    previous_score = float("inf")
    for row in rows:
        candidate_id = row["candidate_id"].strip()
        if not candidate_id:
            raise ValueError("candidate_id cannot be empty")
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)

        score = float(row["score"])
        if score < 0 or score > 100:
            raise ValueError(f"Score out of range for {candidate_id}: {score}")
        if score > previous_score:
            raise ValueError("Scores must be sorted descending")
        previous_score = score

