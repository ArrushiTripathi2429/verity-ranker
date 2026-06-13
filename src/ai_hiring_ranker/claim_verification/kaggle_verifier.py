"""
Kaggle profile verifier — lightweight public profile check.

Uses stdlib HTTP only. Fails gracefully when the profile is private or blocked.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from typing import Optional

from .schemas import EvidenceItem, EvidenceSource
from .github_verifier import _SKILL_GITHUB_SIGNALS

logger = logging.getLogger(__name__)

_KAGGLE_SKILL_KEYWORDS: dict[str, list[str]] = {
    skill: signals.get("keywords", []) + [skill.lower()]
    for skill, signals in _SKILL_GITHUB_SIGNALS.items()
}


def _fetch_profile_text(kaggle_url: str) -> Optional[str]:
    headers = {"User-Agent": "ai-hiring-ranker/1.0"}
    req = urllib.request.Request(kaggle_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace").lower()
    except urllib.error.HTTPError as exc:
        logger.warning("Kaggle HTTP %d for %s", exc.code, kaggle_url)
        return None
    except Exception as exc:
        logger.warning("Kaggle request failed (%s): %s", type(exc).__name__, kaggle_url)
        return None


def _username_from_url(kaggle_url: str) -> Optional[str]:
    match = re.search(r"kaggle\.com/([A-Za-z0-9_.\-]+)", kaggle_url)
    return match.group(1) if match else None


def verify_skills_via_kaggle(
    kaggle_url: str,
    skills: list[str],
    candidate_id: str,
) -> tuple[list[EvidenceItem], list[str]]:
    """Check public Kaggle profile text for skill-related evidence."""
    evidence: list[EvidenceItem] = []
    errors: list[str] = []

    username = _username_from_url(kaggle_url)
    if not username:
        errors.append(f"Could not parse Kaggle username from URL: {kaggle_url}")
        return evidence, errors

    profile_text = _fetch_profile_text(kaggle_url)
    if not profile_text:
        errors.append(f"Kaggle profile unavailable for {username}")
        return evidence, errors

    logger.info("[%s] Kaggle: fetched profile for %s", candidate_id, username)

    for skill in skills:
        keywords = _KAGGLE_SKILL_KEYWORDS.get(skill, [skill.lower()])
        matched = [kw for kw in keywords if kw and kw in profile_text]
        if not matched:
            continue

        evidence.append(
            EvidenceItem(
                source=EvidenceSource.KAGGLE,
                url=kaggle_url,
                skill=skill,
                snippet=(
                    f"Kaggle profile mentions related keywords: "
                    f"{', '.join(matched[:4])}"
                ),
                relevance_score=0.65,
            )
        )

    return evidence, errors
