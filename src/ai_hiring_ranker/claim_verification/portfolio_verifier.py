"""
Portfolio link verifier — checks public portfolio/project URLs for skill evidence.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from typing import Optional

from .schemas import EvidenceItem, EvidenceSource
from .github_verifier import _SKILL_GITHUB_SIGNALS

logger = logging.getLogger(__name__)

_PORTFOLIO_SKILL_KEYWORDS: dict[str, list[str]] = {
    skill: signals.get("keywords", []) + [skill.lower()]
    for skill, signals in _SKILL_GITHUB_SIGNALS.items()
}


def _fetch_url_text(url: str) -> Optional[str]:
    headers = {"User-Agent": "ai-hiring-ranker/1.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace").lower()[:8000]
    except Exception as exc:
        logger.warning("Portfolio request failed (%s): %s", type(exc).__name__, url)
        return None


def verify_skills_via_portfolio(
    portfolio_urls: list[str],
    skills: list[str],
    candidate_id: str,
) -> tuple[list[EvidenceItem], list[str]]:
    """Scan portfolio/project pages for skill-related evidence."""
    evidence: list[EvidenceItem] = []
    errors: list[str] = []

    if not portfolio_urls:
        return evidence, errors

    combined_text = ""
    fetched = 0
    for url in portfolio_urls[:3]:
        text = _fetch_url_text(url)
        if text:
            combined_text += f" {text}"
            fetched += 1
        else:
            errors.append(f"Portfolio URL unavailable: {url}")

    if fetched == 0:
        return evidence, errors

    logger.info("[%s] Portfolio: fetched %d pages", candidate_id, fetched)

    for skill in skills:
        keywords = _PORTFOLIO_SKILL_KEYWORDS.get(skill, [skill.lower()])
        matched = [kw for kw in keywords if kw and kw in combined_text]
        if not matched:
            continue

        evidence.append(
            EvidenceItem(
                source=EvidenceSource.PORTFOLIO,
                url=portfolio_urls[0],
                skill=skill,
                snippet=(
                    f"Portfolio page mentions related keywords: "
                    f"{', '.join(matched[:4])}"
                ),
                relevance_score=0.55,
            )
        )

    return evidence, errors
