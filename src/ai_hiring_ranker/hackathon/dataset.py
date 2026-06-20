"""Load hackathon candidates.jsonl with flexible field names."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


TEXT_FIELDS = ("resume_text", "profile_text", "text", "resume", "bio", "summary")
ID_FIELDS = ("candidate_id", "id", "user_id")
TITLE_FIELDS = ("job_title", "current_title", "title", "headline")
SKILLS_FIELDS = ("skills", "skill_list", "technical_skills")
YEARS_FIELDS = ("years_experience", "experience_years", "total_years_experience", "yoe")
GITHUB_SCORE_FIELDS = ("github_activity_score", "github_score", "github_activity")
ACTIVE_FIELDS = ("last_active_date", "last_active", "last_seen")
RESPONSE_FIELDS = ("recruiter_response_rate", "response_rate", "reply_rate")


def _first(record: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def candidate_id(record: dict[str, Any], line_no: int) -> str:
    value = _first(record, ID_FIELDS)
    if value is None:
        return f"C{line_no:06d}"
    return str(value).strip()


def resume_text(record: dict[str, Any]) -> str:
    text = _first(record, TEXT_FIELDS, "")
    if isinstance(text, str):
        return text.strip()
    return ""


def job_title(record: dict[str, Any]) -> str:
    return str(_first(record, TITLE_FIELDS, "") or "").strip()


def skills_list(record: dict[str, Any]) -> list[str]:
    raw = _first(record, SKILLS_FIELDS, [])
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return []


def years_experience(record: dict[str, Any]) -> Optional[float]:
    raw = _first(record, YEARS_FIELDS)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def github_activity_score(record: dict[str, Any]) -> float:
    raw = _first(record, GITHUB_SCORE_FIELDS, 0.0)
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def days_since_active(record: dict[str, Any], *, now: Optional[datetime] = None) -> Optional[int]:
    raw = _first(record, ACTIVE_FIELDS)
    if not raw:
        return None
    now = now or datetime.now(timezone.utc)
    try:
        if isinstance(raw, (int, float)):
            return int(raw)
        text = str(raw).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, (now - dt).days)
    except Exception:
        return None


def recruiter_response_rate(record: dict[str, Any]) -> Optional[float]:
    raw = _first(record, RESPONSE_FIELDS)
    if raw is None:
        return None
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        return None
    if rate > 1.0:
        rate = rate / 100.0
    return max(0.0, min(1.0, rate))


def iter_candidates(path: Path | str) -> Iterator[tuple[int, dict[str, Any]]]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def count_candidates(path: Path | str) -> int:
    return sum(1 for _ in iter_candidates(path))
