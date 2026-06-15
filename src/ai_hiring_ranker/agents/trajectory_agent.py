"""
Career Trajectory Agent — Layer 9, Agent 3.

Evaluates a candidate's career growth trajectory and seniority fit.

Scoring dimensions:
  - Seniority match:    candidate seniority signal vs. JD required level
  - Career progression: upward title movement across roles
  - Experience depth:   total years, role duration, recency
  - Leadership signals: ownership, mentoring, architecture language
  - Growth velocity:    promotions / meaningful title changes over time
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile
from ..evidence.schemas import CandidateLedger
from ..jd_intelligence.schemas import HiringProfile, SeniorityLevel
from .schemas import AgentRole, AgentVerdict

logger = logging.getLogger(__name__)

# Seniority level → numeric value for distance calculation
_SENIORITY_NUMERIC: dict[SeniorityLevel, float] = {
    SeniorityLevel.INTERN:    0.0,
    SeniorityLevel.JUNIOR:    1.0,
    SeniorityLevel.MID:       2.0,
    SeniorityLevel.SENIOR:    3.0,
    SeniorityLevel.LEAD:      4.0,
    SeniorityLevel.STAFF:     4.5,
    SeniorityLevel.PRINCIPAL: 5.0,
    SeniorityLevel.MANAGER:   4.0,
    SeniorityLevel.UNKNOWN:   2.0,
}

# Expected minimum years for each seniority level
_MIN_YEARS: dict[SeniorityLevel, float] = {
    SeniorityLevel.INTERN:    0.0,
    SeniorityLevel.JUNIOR:    1.0,
    SeniorityLevel.MID:       3.0,
    SeniorityLevel.SENIOR:    5.0,
    SeniorityLevel.LEAD:      6.0,
    SeniorityLevel.STAFF:     7.0,
    SeniorityLevel.PRINCIPAL: 8.0,
    SeniorityLevel.MANAGER:   5.0,
    SeniorityLevel.UNKNOWN:   2.0,
}

# Title keywords → seniority estimate
_TITLE_SENIORITY: dict[str, float] = {
    "intern": 0.0, "trainee": 0.0,
    "junior": 1.0, "associate": 1.0, "jr": 1.0,
    "engineer": 2.0, "developer": 2.0, "analyst": 2.0, "scientist": 2.0,
    "senior": 3.0, "sr": 3.0,
    "lead": 4.0, "tech lead": 4.0,
    "staff": 4.5,
    "principal": 5.0,
    "manager": 4.0, "head": 4.5, "director": 5.0, "vp": 5.5,
}


def _infer_seniority_from_titles(profile: CandidateProfile) -> float:
    """Estimate the candidate's current seniority numeric value from titles."""
    if not profile.career_timeline:
        return 2.0  # neutral default
    # Most recent role first
    for role in profile.career_timeline[:2]:
        title_lower = role.title.lower()
        for keyword, value in sorted(_TITLE_SENIORITY.items(), key=lambda x: -x[1]):
            if keyword in title_lower:
                return value
    return 2.0


def _seniority_match_score(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
) -> float:
    """
    How close is the candidate's seniority to the required level?

    Perfect match (same level) → 1.0
    One level off              → 0.7
    Two levels off             → 0.4
    Three or more levels off   → 0.1
    """
    jd_level = hiring_profile.seniority
    if jd_level == SeniorityLevel.UNKNOWN:
        return 0.6  # can't penalise if JD doesn't specify

    jd_numeric   = _SENIORITY_NUMERIC[jd_level]
    cand_numeric = _infer_seniority_from_titles(profile)

    # Also use the CandidateProfile signal
    cand_signal  = profile.seniority_signal * 5.0  # scale 0–1 → 0–5
    # Blend title-inferred and signal
    cand_est = round(0.6 * cand_numeric + 0.4 * cand_signal, 2)

    distance = abs(jd_numeric - cand_est)
    if distance <= 0.5:
        return 1.0
    if distance <= 1.0:
        return 0.80
    if distance <= 2.0:
        return 0.55
    if distance <= 3.0:
        return 0.30
    return 0.10


def _experience_depth_score(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
) -> float:
    """Score experience depth relative to the JD's minimum years requirement."""
    required_years = hiring_profile.years_of_experience_min or \
                     _MIN_YEARS.get(hiring_profile.seniority, 2.0)

    if profile.total_years_experience is None:
        # Estimate from career timeline
        total = sum(
            r.duration_years or 0.0
            for r in profile.career_timeline
            if r.duration_years is not None
        )
        candidate_years = total if total > 0 else 2.0
    else:
        candidate_years = profile.total_years_experience

    if required_years <= 0:
        ratio = 1.0
    else:
        ratio = candidate_years / required_years

    if ratio >= 1.5:
        return 1.0
    if ratio >= 1.0:
        return 0.85
    if ratio >= 0.75:
        return 0.65
    if ratio >= 0.5:
        return 0.40
    return 0.20


def _career_growth_score(profile: CandidateProfile) -> float:
    """
    Score upward career trajectory.
    Uses the pre-computed career_growth_signal from CandidateProfile +
    leadership signal bonus.
    """
    base = profile.career_growth_signal
    leadership_bonus = profile.leadership_signal * 0.20
    return min(1.0, round(base + leadership_bonus, 4))


def _evaluate_rules(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: Optional[CandidateLedger],
) -> AgentVerdict:
    seniority  = _seniority_match_score(profile, hiring_profile)
    experience = _experience_depth_score(profile, hiring_profile)
    growth     = _career_growth_score(profile)

    score = round(
        0.40 * seniority  +
        0.35 * experience +
        0.25 * growth,
        4,
    )

    flags: list[str] = []
    jd_level = hiring_profile.seniority
    if seniority >= 0.85:
        flags.append(f"Seniority well-matched to {jd_level.value} level")
    elif seniority <= 0.35:
        flags.append(f"Seniority mismatch — required {jd_level.value}")
    if experience >= 0.85:
        flags.append("Meets or exceeds experience requirement")
    elif experience <= 0.40:
        flags.append("Under the minimum years of experience")
    if growth >= 0.70:
        flags.append("Strong upward career trajectory")
    if profile.leadership_signal >= 0.60:
        flags.append("Leadership and ownership language present")

    years_str = (
        f"{profile.total_years_experience:.1f}" if profile.total_years_experience
        else "estimated from timeline"
    )
    reasoning = (
        f"Seniority match score: {seniority:.2f} vs required '{jd_level.value}'. "
        f"Experience depth: {experience:.2f} ({years_str} years total). "
        f"Career growth signal: {growth:.2f} (trajectory + leadership). "
        f"Leadership signal: {profile.leadership_signal:.2f}."
    )

    evidence = [
        f"{r.title} at {r.company} ({r.start_year or '?'}–{r.end_year or 'present'})"
        for r in profile.career_timeline[:3]
    ]

    return AgentVerdict(
        agent=AgentRole.TRAJECTORY,
        candidate_id=profile.candidate_id,
        score=score,
        reasoning=reasoning,
        flags=flags,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# LLM scoring
# ---------------------------------------------------------------------------


def _evaluate_llm(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: Optional[CandidateLedger],
) -> AgentVerdict:
    from ..llm_provider import chat_completion
    import json

    timeline_text = "; ".join(
        f"{r.title} at {r.company} "
        f"({r.start_year or '?'}–{r.end_year or 'now'}, "
        f"{r.duration_years or '?'}yrs)"
        for r in profile.career_timeline[:5]
    ) or "no timeline extracted"

    system_prompt = (
        "You are a Career Trajectory Agent. Evaluate the candidate's seniority fit and "
        "career growth. Return ONLY JSON with keys: score (float 0-1), "
        "reasoning (string, cite specific roles/years), flags (list), evidence (list)."
    )
    user_prompt = f"""Job: {hiring_profile.job_title}
Required seniority: {hiring_profile.seniority.value}
Min experience: {hiring_profile.years_of_experience_min or 'not specified'} years

Candidate: {profile.name or profile.candidate_id}
Career timeline: {timeline_text}
Total experience: {profile.total_years_experience or 'unknown'} years
Seniority signal: {profile.seniority_signal:.2f}
Leadership signal: {profile.leadership_signal:.2f}
Career growth signal: {profile.career_growth_signal:.2f}

Evaluate career trajectory and seniority fit. Return JSON."""

    raw = chat_completion(system_prompt, user_prompt)
    try:
        data = json.loads(raw.strip().strip("```json").strip("```"))
        return AgentVerdict(
            agent=AgentRole.TRAJECTORY,
            candidate_id=profile.candidate_id,
            score=float(data.get("score", 0.5)),
            reasoning=str(data.get("reasoning", "")),
            flags=list(data.get("flags", [])),
            evidence=list(data.get("evidence", [])),
        )
    except Exception as exc:
        logger.warning("Trajectory LLM parse failed (%s) — using rules.", exc)
        return _evaluate_rules(profile, hiring_profile, ledger)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_trajectory(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: Optional[CandidateLedger] = None,
    *,
    force_fallback: bool = False,
) -> AgentVerdict:
    """
    Run the Career Trajectory Agent for one candidate.

    Args:
        profile:        CandidateProfile from Layer 4.
        hiring_profile: HiringProfile from Layer 2.
        ledger:         CandidateLedger from Layer 6 (optional).
        force_fallback: Always use rule-based evaluation.

    Returns:
        AgentVerdict with seniority match, experience depth, and growth score.
    """
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))
    if force_fallback or not has_key:
        return _evaluate_rules(profile, hiring_profile, ledger)
    try:
        return _evaluate_llm(profile, hiring_profile, ledger)
    except Exception as exc:
        logger.warning("Trajectory LLM failed (%s) — using rules.", exc)
        return _evaluate_rules(profile, hiring_profile, ledger)
