"""
Rule-based claim verifier — works entirely from resume text and Layer 4 data.

Used when:
  - No GitHub URL is available.
  - GitHub API is unavailable or rate-limited.
  - force_fallback=True (testing / offline mode).

Strategy per skill:
  1. Count evidence_snippets from SkillClaim (Layer 4 already extracted these).
  2. Check for production / deployment language in the snippets.
  3. Check for metric / quantified evidence.
  4. Check for recency signals (years mentioned, "current", "recent").
  5. Apply negation check — if the resume denies the skill, mark UNSUPPORTED.
  6. Map evidence strength → VerificationStatus + confidence.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile, SkillClaim, SkillConfidence
from ..config import VerificationConfig, get_verification_config
from ..ingestion.schemas import VerificationStatus
from .schemas import EvidenceItem, EvidenceSource, VerifiedClaim, VerificationReport
from .utils import status_from_confidence

_CURRENT_YEAR = datetime.now().year

_NEGATION_RE = re.compile(
    r"\b(no |not |without |lack(ing)?|limited |never |haven.t |don.t have )\b",
    re.I,
)
_PRODUCTION_RE = re.compile(
    r"\b(production|deployed|live|served|released|launched|shipped|ran in|cloud)\b",
    re.I,
)
_METRIC_RE = re.compile(
    r"\d+\s*(%|x\b|ms\b|users?\b|requests?\b|latency|throughput|accuracy|precision|recall)",
    re.I,
)
_RECENCY_RE = re.compile(
    r"\b(current(ly)?|recent(ly)?|now|today|202[0-9]|ongoing)\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b(20[0-2]\d|19[89]\d)\b")


def _snippets_are_recent(
    snippets: list[str],
    recency_cutoff_years: int = 3,
) -> bool:
    for s in snippets:
        years = [int(y) for y in _YEAR_RE.findall(s)]
        if years and max(years) >= _CURRENT_YEAR - recency_cutoff_years:
            return True
        if _RECENCY_RE.search(s):
            return True
    return False


def _has_production_evidence(snippets: list[str]) -> bool:
    return any(_PRODUCTION_RE.search(s) for s in snippets)


def _has_metric(snippets: list[str]) -> bool:
    return any(_METRIC_RE.search(s) for s in snippets)


def _is_negated(snippets: list[str]) -> bool:
    """Return True if any snippet denies the skill."""
    return any(_NEGATION_RE.search(s) for s in snippets)


def _compute_status_and_confidence(
    skill_claim: SkillClaim,
    negated: bool,
    config: Optional[VerificationConfig] = None,
) -> tuple[VerificationStatus, float, str]:
    """
    Determine VerificationStatus + confidence + reasoning from rule signals.

    Returns: (status, confidence, reasoning)
    """
    cfg = config or get_verification_config()
    recency_cutoff = cfg.github.recency_cutoff_years
    snippets = skill_claim.evidence_snippets
    n_snippets = len(snippets)
    conf = skill_claim.confidence

    if negated:
        return (
            VerificationStatus.UNSUPPORTED,
            0.0,
            "Resume explicitly denies this skill in a negation sentence.",
        )

    if n_snippets == 0 and conf == SkillConfidence.WEAK:
        return (
            VerificationStatus.UNSUPPORTED,
            0.05,
            "Skill mentioned but no supporting context found in resume.",
        )

    recent = _snippets_are_recent(snippets, recency_cutoff)
    prod = _has_production_evidence(snippets)
    metric = _has_metric(snippets)
    explicit = conf == SkillConfidence.EXPLICIT

    score = 0.0
    reasons: list[str] = []

    if explicit:
        score += 0.35
        reasons.append("explicitly listed in Skills section or named directly")
    elif conf == SkillConfidence.INFERRED:
        score += 0.20
        reasons.append("inferred from project/context language")
    else:
        score += 0.10
        reasons.append("weakly mentioned")

    if n_snippets >= 2:
        score += 0.20
        reasons.append(f"{n_snippets} supporting evidence sentences")
    elif n_snippets == 1:
        score += 0.10
        reasons.append("1 supporting evidence sentence")

    if prod:
        score += 0.20
        reasons.append("production/deployment evidence present")

    if metric:
        score += 0.10
        reasons.append("quantified achievement present")

    if recent:
        score += 0.10
        reasons.append(f"evidence is recent (last {recency_cutoff} years)")
    else:
        score = max(score - 0.10, 0.0)
        reasons.append(f"evidence may be older than {recency_cutoff} years")

    score = min(round(score, 2), 1.0)
    status = status_from_confidence(score, cfg.verification_labels)
    reasoning = "; ".join(reasons) + f" → score={score:.2f}"
    return status, score, reasoning


def verify_candidate_rules(
    profile: CandidateProfile,
    config: Optional[VerificationConfig] = None,
) -> VerificationReport:
    """
    Run rule-based verification on all skill claims in a CandidateProfile.

    No external API calls. Uses only evidence_snippets from Layer 4.

    Returns:
        VerificationReport with one VerifiedClaim per skill.
    """
    verified_claims: list[VerifiedClaim] = []

    for skill_claim in profile.skills:
        snippets = skill_claim.evidence_snippets
        negated = _is_negated(snippets)

        status, confidence, reasoning = _compute_status_and_confidence(
            skill_claim,
            negated,
            config=config,
        )

        evidence_items: list[EvidenceItem] = []
        for snippet in snippets[:3]:
            evidence_items.append(
                EvidenceItem(
                    source=EvidenceSource.RESUME,
                    skill=skill_claim.skill,
                    snippet=snippet,
                    relevance_score=confidence,
                )
            )

        claim_text = snippets[0] if snippets else ""

        verified_claims.append(
            VerifiedClaim(
                candidate_id=profile.candidate_id,
                skill=skill_claim.skill,
                claim_text=claim_text,
                status=status,
                confidence=confidence,
                evidence=evidence_items,
                reasoning=reasoning,
            )
        )

    return VerificationReport(
        candidate_id=profile.candidate_id,
        candidate_name=profile.name,
        github_url=None,
        claims=verified_claims,
        github_checked=False,
    )
