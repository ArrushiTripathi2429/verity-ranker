"""
JD Fit Agent — Layer 9, Agent 1.

Evaluates how well a candidate's skills and background match the
job description's stated requirements.

Scoring logic:
  - Required skill coverage (proportion of required skills present)
  - Preferred skill coverage (bonus)
  - Hidden expectation alignment (partial signal from profile signals)
  - Domain/industry match

Uses the Layer 7 graph so "sklearn" counts toward "Machine Learning",
and "Flask" gives partial credit when "FastAPI" is required.

Two modes:
  LLM   — rich, cited reasoning via structured_completion().
  Rules — deterministic formula-based scoring (no API key needed).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile
from ..evidence.schemas import CandidateLedger
from ..jd_intelligence.schemas import HiringProfile
from ..skill_graph.graph import expand_skills
from .schemas import AgentRole, AgentVerdict

logger = logging.getLogger(__name__)

# Weight for required vs preferred skill coverage
_REQ_WEIGHT  = 0.70
_PREF_WEIGHT = 0.30


# ---------------------------------------------------------------------------
# Rule-based scoring
# ---------------------------------------------------------------------------


def _skill_coverage(
    candidate_skills: list[str],
    jd_skills: list[str],
    *,
    max_hops: int = 1,
    min_weight: float = 0.40,
) -> tuple[float, list[str], list[str]]:
    """
    Compute weighted skill coverage via graph expansion.

    Returns:
        (coverage_score 0–1, exact_matches, partial_matches)
    """
    if not jd_skills:
        return 1.0, [], []

    jd_expanded = expand_skills(jd_skills, max_hops=max_hops, min_weight=min_weight)
    cand_canonical = {s.strip().title() for s in candidate_skills}

    # Also expand the candidate side for synonym resolution
    cand_exp = expand_skills(candidate_skills, max_hops=1, min_weight=0.70)
    cand_expanded = {e.canonical for e in cand_exp.expanded_skills}

    total_weight    = 0.0
    achieved_weight = 0.0
    exact_matches:   list[str] = []
    partial_matches: list[str] = []

    for exp in jd_expanded.expanded_skills:
        total_weight += exp.weight
        if exp.canonical in cand_canonical or exp.canonical in cand_expanded:
            achieved_weight += exp.weight
            if exp.hop_distance == 0:
                exact_matches.append(exp.canonical)
            else:
                partial_matches.append(exp.canonical)

    score = achieved_weight / total_weight if total_weight > 0 else 0.0
    return round(min(1.0, score), 4), list(dict.fromkeys(exact_matches)), list(dict.fromkeys(partial_matches))


def _domain_match(profile: CandidateProfile, hiring_profile: HiringProfile) -> float:
    """Estimate domain alignment from career timeline and project descriptions."""
    if not hiring_profile.domain:
        return 0.5  # neutral when JD has no domain

    domain_lower = hiring_profile.domain.lower()
    text_parts: list[str] = []
    for role in profile.career_timeline:
        text_parts.append(role.title.lower())
        text_parts.append(role.company.lower())
    for proj in profile.projects:
        text_parts.append(proj.description.lower())

    combined = " ".join(text_parts)
    return 0.8 if domain_lower in combined else 0.3


def _evaluate_rules(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: Optional[CandidateLedger],
) -> AgentVerdict:
    req_score, req_exact, req_partial = _skill_coverage(
        profile.skill_names, hiring_profile.all_required_skill_names
    )
    pref_score, pref_exact, pref_partial = _skill_coverage(
        profile.skill_names, hiring_profile.all_preferred_skill_names, min_weight=0.50
    )
    domain_score = _domain_match(profile, hiring_profile)

    combined = (
        _REQ_WEIGHT  * req_score +
        _PREF_WEIGHT * pref_score
    )
    # Domain nudge: ±0.05
    combined = min(1.0, combined + (domain_score - 0.5) * 0.10)

    # Build flags
    flags: list[str] = []
    missing_required = [
        s for s in hiring_profile.all_required_skill_names
        if s not in req_exact and s not in req_partial
    ]
    if missing_required:
        flags.append(f"Missing required: {', '.join(missing_required[:4])}")
    if req_partial:
        flags.append(f"Partial via graph: {', '.join(req_partial[:3])}")
    if pref_score >= 0.6:
        flags.append("Strong preferred skill coverage")

    # Build reasoning
    reasoning_parts = [
        f"Required skill coverage: {req_score:.0%} "
        f"({len(req_exact)} exact, {len(req_partial)} via graph expansion).",
    ]
    if pref_exact or pref_partial:
        reasoning_parts.append(
            f"Preferred skill coverage: {pref_score:.0%} "
            f"({len(pref_exact)} exact, {len(pref_partial)} via graph)."
        )
    if missing_required:
        reasoning_parts.append(
            f"Notable gaps in required skills: {', '.join(missing_required[:3])}."
        )
    if hiring_profile.domain:
        reasoning_parts.append(
            f"Domain match ({hiring_profile.domain}): {'aligned' if domain_score >= 0.7 else 'unclear'}."
        )

    evidence = [s for skill in req_exact[:3]
                for sc in profile.skills if sc.skill == skill
                for s in sc.evidence_snippets[:1]]

    return AgentVerdict(
        agent=AgentRole.JD_FIT,
        candidate_id=profile.candidate_id,
        score=round(combined, 4),
        reasoning=" ".join(reasoning_parts),
        flags=flags,
        evidence=evidence[:5],
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

    required_skills  = ", ".join(hiring_profile.all_required_skill_names) or "none"
    preferred_skills = ", ".join(hiring_profile.all_preferred_skill_names) or "none"
    candidate_skills = ", ".join(profile.skill_names[:20]) or "none listed"
    projects_text    = "; ".join(
        f"{p.title}: {p.description[:80]}" for p in profile.projects[:3]
    ) or "none"

    system_prompt = (
        "You are a JD Fit Agent. Evaluate how well the candidate matches the job description. "
        "Return ONLY a JSON object with keys: score (float 0-1), reasoning (string, 3-5 sentences, cite evidence), "
        "flags (list of short strings), evidence (list of resume snippets). "
        "Be strict: missing required skills must reduce the score significantly."
    )
    user_prompt = f"""Job: {hiring_profile.job_title} ({hiring_profile.seniority.value})
Required skills: {required_skills}
Preferred skills: {preferred_skills}
Domain: {hiring_profile.domain or 'not specified'}

Candidate: {profile.name or profile.candidate_id}
Skills: {candidate_skills}
Projects: {projects_text}
Seniority signal: {profile.seniority_signal:.2f}

Evaluate JD fit. Return JSON."""

    import json
    raw = chat_completion(system_prompt, user_prompt)
    try:
        data = json.loads(raw.strip().strip("```json").strip("```"))
        return AgentVerdict(
            agent=AgentRole.JD_FIT,
            candidate_id=profile.candidate_id,
            score=float(data.get("score", 0.5)),
            reasoning=str(data.get("reasoning", "")),
            flags=list(data.get("flags", [])),
            evidence=list(data.get("evidence", [])),
        )
    except Exception as exc:
        logger.warning("JD Fit LLM parse failed (%s) — falling back to rules.", exc)
        return _evaluate_rules(profile, hiring_profile, ledger)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_jd_fit(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: Optional[CandidateLedger] = None,
    *,
    force_fallback: bool = False,
) -> AgentVerdict:
    """
    Run the JD Fit Agent for one candidate.

    Args:
        profile:        CandidateProfile from Layer 4.
        hiring_profile: HiringProfile from Layer 2.
        ledger:         CandidateLedger from Layer 6 (optional, adds proof context).
        force_fallback: Always use rule-based evaluation.

    Returns:
        AgentVerdict with score, reasoning, flags, and evidence.
    """
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))
    if force_fallback or not has_key:
        return _evaluate_rules(profile, hiring_profile, ledger)
    try:
        return _evaluate_llm(profile, hiring_profile, ledger)
    except Exception as exc:
        logger.warning("JD Fit LLM failed (%s) — using rules.", exc)
        return _evaluate_rules(profile, hiring_profile, ledger)
