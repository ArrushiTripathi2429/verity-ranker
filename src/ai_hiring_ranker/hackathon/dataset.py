"""Load hackathon candidates.jsonl using the official Redrob schema, with a
flexible-field fallback for resilience against minor format drift."""

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
    """Build a text blob from the nested profile + career history descriptions
    so regex skill/evidence extraction has real content to scan."""
    profile = record.get("profile", {}) or {}
    parts: list[str] = []
    if profile.get("headline"):
        parts.append(str(profile["headline"]))
    if profile.get("summary"):
        parts.append(str(profile["summary"]))
    for role in record.get("career_history", []) or []:
        desc = role.get("description")
        if desc:
            parts.append(f"{role.get('title', '')} at {role.get('company', '')}: {desc}")

    if parts:
        return " ".join(parts).strip()

    # Fallback for unexpected formats
    text = _first(record, TEXT_FIELDS, "")
    return text.strip() if isinstance(text, str) else ""


def job_title(record: dict[str, Any]) -> str:
    profile = record.get("profile", {}) or {}
    title = profile.get("current_title") or profile.get("headline")
    if title:
        return str(title).strip()
    return str(_first(record, TITLE_FIELDS, "") or "").strip()


def skills_list(record: dict[str, Any]) -> list[str]:
    raw = record.get("skills")
    if isinstance(raw, list):
        names: list[str] = []
        for s in raw:
            if isinstance(s, dict):
                name = s.get("name")
                if name:
                    names.append(str(name).strip())
            elif isinstance(s, str) and s.strip():
                names.append(s.strip())
        if names:
            return names

    # Fallback for unexpected formats (string-list, alt field names)
    raw2 = _first(record, SKILLS_FIELDS, [])
    if isinstance(raw2, str):
        return [s.strip() for s in raw2.split(",") if s.strip()]
    if isinstance(raw2, list):
        return [str(s).strip() for s in raw2 if str(s).strip()]
    return []


def years_experience(record: dict[str, Any]) -> Optional[float]:
    profile = record.get("profile", {}) or {}
    raw = profile.get("years_of_experience")
    if raw is None:
        raw = _first(record, YEARS_FIELDS)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def github_activity_score(record: dict[str, Any]) -> float:
    signals = record.get("redrob_signals", {}) or {}
    raw = signals.get("github_activity_score")
    if raw is None:
        raw = _first(record, GITHUB_SCORE_FIELDS, 0.0)
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if score < 0:
        return 0.0  # -1 in the dataset means "no GitHub linked" — treat as no evidence
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def days_since_active(record: dict[str, Any], *, now: Optional[datetime] = None) -> Optional[int]:
    signals = record.get("redrob_signals", {}) or {}
    raw = signals.get("last_active_date")
    if not raw:
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
    signals = record.get("redrob_signals", {}) or {}
    raw = signals.get("recruiter_response_rate")
    if raw is None:
        raw = _first(record, RESPONSE_FIELDS)
    if raw is None:
        return None
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        return None
    if rate < 0:
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