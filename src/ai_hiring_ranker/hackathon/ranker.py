"""Fast offline top-100 selection and CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .reasoning import build_reasoning
from .schemas import SubmissionRow


def load_feature_cache(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rank_candidates(
    features: Iterable[dict[str, Any]],
    *,
    job_title: str,
    top_k: int = 100,
) -> list[SubmissionRow]:
    """
    Sort candidates by base_score descending, tie-break candidate_id ascending.
    Return exactly top_k submission rows with reasoning.
    """
    ranked = sorted(
        list(features),
        key=lambda row: (-float(row.get("base_score", 0.0)), str(row.get("candidate_id", ""))),
    )[:top_k]

    output: list[SubmissionRow] = []
    for idx, row in enumerate(ranked, start=1):
        score = round(float(row.get("base_score", 0.0)), 2)
        output.append(
            SubmissionRow(
                candidate_id=str(row["candidate_id"]),
                rank=idx,
                score=score,
                reasoning=build_reasoning(row, job_title, idx),
            )
        )

    # Enforce non-increasing scores after tie-break ordering
    for i in range(1, len(output)):
        if output[i].score > output[i - 1].score:
            output[i].score = output[i - 1].score

    return output


def write_submission_csv(rows: list[SubmissionRow], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row.candidate_id,
                    "rank": row.rank,
                    "score": f"{row.score:.2f}",
                    "reasoning": row.reasoning,
                }
            )
    return path
