"""
Technical Fit Agent — Layer 9, Agent 2.

Evaluates the depth and breadth of a candidate's technical skills,
going beyond just "does the skill appear" to "how strong is the evidence".

Scoring dimensions:
  - Skill breadth:  how many required/preferred skills are covered
  - Skill depth:    evidence snippets count, years of experience per skill
  - Production signals: deployed/shipped/served language in profile
  - Project quality: production-grade projects with measurable outcomes
  - Verification strength: how many technical claims are verified (from ledger)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile, SkillConfidence
from ..evidence.schemas import CandidateLedger
from ..jd_intelligence.schemas import HiringProfile
from ..skill_graph.graph import expand_skills
from .schemas import AgentRole, AgentVerdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based helpers
# ---------------------------------------------------------------------------


def _skill_depth_score(profile: CandidateProfile, jd_skill_names: list[str]) -> float:
    """
    Score technical depth based on evidence snippets and experience years.
    Returns 0–1.
    """
    if not profile.skills:
        return 0.0

    relevant = [
        s for s in profile.skills
        if s.skill in {sk.strip().title() for sk in jd_skill_names}
    ]
    if not relevant:
        relevant = profile.skills  # fall back to all skills

    depth_scores: list[float] = []
    for skill in relevant:
        score = 0.0
        # Evidence snippet presence
        score += min(len(skill.evidence_snippets) * 0.20, 0.40)
        # Explicit vs weak confidence
        if skill.confidence == SkillConfidence.EXPLICIT:
            score += 0.30
        elif skill.confidence == SkillConfidence.INFERRED:
            score += 0.15
        # Years of experience
        if skill.years_of_experience:
            score += min(skill.years_of_experience * 0.05, 0.30)
        depth_scores.append(min(1.0, score))

    return round(sum(depth_scores) / len(depth_scores), 4) if depth_scores else 0.0


def _production_score(profile: CandidateProfile) -> float:
    """
    Score production-grade engineering evidence (0–1).
    Uses the pre-computed signal from CandidateProfile.
    """
    # production_signal (0–1) + project quality bonus
    prod_bonus = sum(
        0.1 for p in profile.projects
        if p.is_production and p.has_metrics
    )
    return min(1.0, profile.production_signal + prod_bonus)


def _verified_technical_ratio(
    profile: CandidateProfile,
    ledger: Optional[CandidateLedger],
    jd_skills: list[str],
) -> float:
    """
    Fraction of technical JD skills that are verified in the ledger.
    Falls back to resume-only signal if ledger is absent.
    """
    if not ledger or not jd_skills:
        # Fallback: use production + achievement signals as a proxy
        return min(1.0, (profile.production_signal + profile.achievement_signal) / 2)

    jd_set = {s.strip().title() for s in jd_skills}
    relevant_entries = [e for e in ledger.entries if e.skill in jd_set]
    if not relevant_entries:
        return 0.5  # neutral — no relevant claims in ledger

    from ..ingestion.schemas import VerificationStatus
    weights = {
        VerificationStatus.VERIFIED:    1.0,
        VerificationStatus.WEAK:        0.6,
        VerificationStatus.INFERRED:    0.3,
        VerificationStatus.UNSUPPORTED: 0.0,
        VerificationStatus.PENDING:     0.2,
    }
    total = sum(weights.get(e.verification_status, 0.0) for e in relevant_entries)
    return round(total / len(relevant_entries), 4)


def _evaluate_rules(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: Optional[CandidateLedger],
) -> AgentVerdict:
    all_jd_skills = hiring_profile.all_skill_names

    # Breadth: how many JD skills covered (with graph expansion)
    jd_exp = expand_skills(all_jd_skills, max_hops=1, min_weight=0.45)
    cand_set = {s.strip().title() for s in profile.skill_names}
    cand_exp = expand_skills(profile.skill_names, max_hops=1, min_weight=0.70)
    cand_expanded = {e.canonical for e in cand_exp.expanded_skills}

    covered = sum(
        1 for e in jd_exp.expanded_skills
        if e.canonical in cand_set or e.canonical in cand_expanded
    )
    breadth = covered / len(jd_exp.expanded_skills) if jd_exp.expanded_skills else 0.5

    depth      = _skill_depth_score(profile, all_jd_skills)
    production = _production_score(profile)
    verified   = _verified_technical_ratio(profile, ledger, all_jd_skills)

    # Weighted blend
    score = round(
        0.35 * breadth   +
        0.30 * depth     +
        0.20 * production +
        0.15 * verified,
        4,
    )

    flags: list[str] = []
    if production >= 0.7:
        flags.append("Strong production engineering evidence")
    if depth < 0.30:
        flags.append("Low skill depth — mostly surface-level mentions")
    if verified >= 0.75:
        flags.append("Most technical claims are verified")
    elif verified < 0.25:
        flags.append("Few technical claims have external verification")
    if profile.achievement_signal >= 0.6:
        flags.append("Measurable technical achievements present")

    reasoning = (
        f"Skill breadth: {breadth:.0%} of JD skills covered (including graph expansion). "
        f"Skill depth score: {depth:.2f} based on evidence snippet density and experience years. "
        f"Production signal: {production:.2f} (deployed/shipped/served language and production projects). "
        f"Verified technical claim ratio: {verified:.2f}."
    )

    evidence = [
        snip
        for skill in profile.skills[:5]
        for snip in skill.evidence_snippets[:1]
    ]

    return AgentVerdict(
        agent=AgentRole.TECHNICAL_FIT,
        candidate_id=profile.candidate_id,
        score=min(1.0, score),
        reasoning=reasoning,
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
    import json

    skills_text = "; ".join(
        f"{s.skill} ({s.confidence.value}, {len(s.evidence_snippets)} snippets)"
        for s in profile.skills[:15]
    ) or "none"
    projects_text = "; ".join(
        f"{p.title} [prod={p.is_production}, metrics={p.has_metrics}]"
        for p in profile.projects[:4]
    ) or "none"
    verified_skills = ", ".join(ledger.verified_skills[:8]) if ledger else "ledger unavailable"

    system_prompt = (
        "You are a Technical Fit Agent. Evaluate the technical depth and breadth of a candidate "
        "against a job description. Return ONLY JSON with keys: score (float 0-1), "
        "reasoning (string, 3-5 sentences citing specific evidence), "
        "flags (list), evidence (list of resume snippets)."
    )
    user_prompt = f"""Job: {hiring_profile.job_title}
Required: {', '.join(hiring_profile.all_required_skill_names)}
Preferred: {', '.join(hiring_profile.all_preferred_skill_names)}

Candidate: {profile.name or profile.candidate_id}
Skills (confidence, evidence count): {skills_text}
Projects: {projects_text}
Verified skills (from evidence ledger): {verified_skills}
Production signal: {profile.production_signal:.2f}
Achievement signal: {profile.achievement_signal:.2f}

Evaluate technical fit. Return JSON."""

    raw = chat_completion(system_prompt, user_prompt)
    try:
        data = json.loads(raw.strip().strip("```json").strip("```"))
        return AgentVerdict(
            agent=AgentRole.TECHNICAL_FIT,
            candidate_id=profile.candidate_id,
            score=float(data.get("score", 0.5)),
            reasoning=str(data.get("reasoning", "")),
            flags=list(data.get("flags", [])),
            evidence=list(data.get("evidence", [])),
        )
    except Exception as exc:
        logger.warning("Technical Fit LLM parse failed (%s) — using rules.", exc)
        return _evaluate_rules(profile, hiring_profile, ledger)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_technical_fit(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: Optional[CandidateLedger] = None,
    *,
    force_fallback: bool = False,
) -> AgentVerdict:
    """
    Run the Technical Fit Agent for one candidate.

    Args:
        profile:        CandidateProfile from Layer 4.
        hiring_profile: HiringProfile from Layer 2.
        ledger:         CandidateLedger from Layer 6 (optional).
        force_fallback: Always use rule-based evaluation.

    Returns:
        AgentVerdict with technical depth/breadth score and evidence.
    """
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))
    if force_fallback or not has_key:
        return _evaluate_rules(profile, hiring_profile, ledger)
    try:
        return _evaluate_llm(profile, hiring_profile, ledger)
    except Exception as exc:
        logger.warning("Technical Fit LLM failed (%s) — using rules.", exc)
        return _evaluate_rules(profile, hiring_profile, ledger)
