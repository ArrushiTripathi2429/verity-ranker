"""Generate rank-appropriate, fact-grounded reasoning strings."""

from __future__ import annotations

from typing import Any


def _clean(items: list[str]) -> list[str]:
    return [x.strip() for x in items if x and x.strip()]


def build_reasoning(
    features: dict[str, Any],
    hiring_profile_title: str,
    rank: int,
) -> str:
    """
    Build one specific reasoning line using only cached profile facts.

    Tone scales with rank — top ranks are confident, lower ranks are cautious.
    """
    cid = features.get("candidate_id", "")
    title = features.get("job_title") or features.get("candidate_name") or cid
    years = features.get("years_experience")
    matched = _clean(features.get("matched_required", []))
    partial = _clean(features.get("partial_matches", []))
    missing = _clean(features.get("missing_required", []))
    gh = features.get("github_activity_score", 0.0)
    engagement = features.get("engagement_multiplier", 1.0)
    honeypot = features.get("honeypot_risk", 0.0)
    stuffer = features.get("stuffer_risk", 0.0)

    parts: list[str] = []

    if rank <= 10:
        opener = f"Strong fit for {hiring_profile_title}:"
    elif rank <= 30:
        opener = f"Solid match for {hiring_profile_title}:"
    elif rank <= 60:
        opener = f"Moderate fit for {hiring_profile_title}:"
    else:
        opener = f"Marginal fit for {hiring_profile_title}:"

    parts.append(opener)

    if matched:
        parts.append(f"covers required skills {', '.join(matched[:5])}")
    if partial:
        parts.append(f"adjacent experience in {', '.join(partial[:3])}")
    if years is not None:
        parts.append(f"{years:.0f} years experience as {title}")
    elif title:
        parts.append(f"current title {title}")

    if gh >= 0.55:
        parts.append(f"github_activity_score {gh:.2f} supports technical activity")
    elif gh <= 0.10 and matched:
        parts.append("limited GitHub activity despite listed engineering skills")

    if missing and rank <= 50:
        parts.append(f"gap on {', '.join(missing[:2])}")

    if engagement < 0.85:
        parts.append("down-ranked for low recruiter response / inactivity on platform")

    if honeypot >= 0.25:
        parts.append("profile flagged for suspicious or inconsistent claims")
    elif stuffer >= 0.25:
        parts.append("profile shows keyword-stuffing patterns versus role depth")

    if rank > 70 and not missing and not matched:
        parts.append("limited direct evidence for core JD requirements")

    sentence = " ".join(parts)
    if len(sentence) > 480:
        sentence = sentence[:477] + "..."
    return sentence
