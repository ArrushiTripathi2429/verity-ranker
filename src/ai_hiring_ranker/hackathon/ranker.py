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


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Null-safe float coercion. Treats missing key AND explicit None the same way."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Null-safe int coercion."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    """Null-safe bool coercion."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _score_value(row: dict[str, Any]) -> float:
    val = row.get("final_score")
    if val is None:
        val = row.get("base_score")
    return _safe_float(val, 0.0)


def _listwise_sort_key(row: dict[str, Any]) -> tuple:
    """
    Rule-based listwise tie-break (Layer 11 offline).

    Uses full-precision score plus cached dimensions + location + disqualifiers.
    """
    dims = row.get("dimensions") or {}
    
    # Location matching (JD prefers Pune/Noida)
    location = (row.get("location") or "").lower()
    PREFERRED_LOCATIONS = {"pune", "noida", "delhi", "ncr", "gurgaon", "gurugram", "hyderabad"}
    location_penalty = 0 if any(city in location for city in PREFERRED_LOCATIONS) else 1
    
    # Disqualifiers: if any critical issue exists, push to bottom
    disqualifiers = row.get("disqualifiers") or []
    has_critical_disqualifier = len(disqualifiers) > 0
    
    # Open to work (JD wants candidates actively seeking)
    open_to_work = _safe_bool(row.get("open_to_work"), True)  # Default True if missing
    
    # Behavioral signals: offer acceptance rate (low = flight risk)
    offer_acceptance = _safe_float(row.get("offer_acceptance_rate"), 1.0)
    
    # Notice period: prefer short notice
    notice_days = _safe_int(row.get("notice_period_days"), 30)
    
    # Response time: prefer fast responders
    response_time = _safe_float(row.get("avg_response_time_hours"), 999)
    
    return (
        # Primary: score (descending)
        -_score_value(row),
        # Secondary: no critical disqualifiers
        has_critical_disqualifier,
        # Tertiary: proof strength dimension
        -_safe_float(dims.get("proof_strength"), 0.0),
        # Skill fit
        -_safe_float(dims.get("skill_fit"), 0.0),
        # Behavioral: open to work
        open_to_work,  # True sorts before False
        # Behavioral: notice period (prefer short)
        notice_days,
        # Behavioral: response time (prefer fast)
        response_time,
        # Behavioral: offer acceptance rate (higher is better)
        -offer_acceptance,
        # Location preference
        location_penalty,
        # Seniority match
        -_safe_float(dims.get("seniority_match"), 0.0),
        # Career growth
        -_safe_float(dims.get("career_growth"), 0.0),
        # Experience depth
        -_safe_float(dims.get("experience_depth"), 0.0),
        # GitHub activity
        -_safe_float(row.get("github_activity_score"), 0.0),
        # Tie-break by ID
        str(row.get("candidate_id", "")),
    )


def _submission_sort_key(row: dict[str, Any]) -> tuple:
    """Final ordering for CSV: rounded score desc, candidate_id asc on ties."""
    return (-round(_score_value(row), 2), str(row.get("candidate_id", "")))


def rank_candidates(
    features: Iterable[dict[str, Any]],
    *,
    job_title: str,
    top_k: int = 100,
    listwise_pool: int = 300,
) -> list[SubmissionRow]:
    """
    Sort candidates and return exactly ``top_k`` submission rows.

    1. Listwise re-rank a shortlist pool (Layer 11, offline rules with location + disqualifiers).
    2. Re-order by displayed score + candidate_id tie-break for validator compliance.
    3. Assign ranks 1..top_k with fact-grounded, JD-specific reasoning.
    """
    all_rows = list(features)
    pool_size = max(top_k, listwise_pool)
    pool = sorted(all_rows, key=_listwise_sort_key)[:pool_size]
    ranked = sorted(pool[:top_k], key=_submission_sort_key)

    output: list[SubmissionRow] = []
    for idx, row in enumerate(ranked, start=1):
        score = round(_score_value(row), 2)
        output.append(
            SubmissionRow(
                candidate_id=str(row["candidate_id"]),
                rank=idx,
                score=score,
                reasoning=build_reasoning(row, job_title, idx),
            )
        )

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