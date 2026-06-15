"""
Verification Agent — Layer 9, Agent 4.

Reads the Evidence Ledger (Layer 6) and scores how trustworthy the
candidate's profile is based on evidence backing.

Scoring dimensions:
  - Proof strength:       overall verified/weak/inferred/unsupported ratio
  - Critical skill cover: are the most-required JD skills verified?
  - Claim density:        total verified claims relative to skill count
  - Red flags:            contradicted or completely unsupported skills
  - Platform activity:    GitHub/Kaggle/portfolio links checked

If no ledger is available, falls back to resume-only signals
(production_signal + achievement_signal) as a proxy.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile
from ..evidence.schemas import CandidateLedger
from ..ingestion.schemas import VerificationStatus
from ..jd_intelligence.schemas import HiringProfile
from .schemas import AgentRole, AgentVerdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based helpers
# ---------------------------------------------------------------------------


def _critical_skill_verification(
    ledger: CandidateLedger,
    required_skills: list[str],
) -> float:
    """
    What fraction of required JD skills are verified or weakly verified?
    Returns 0–1.
    """
    if not required_skills:
        return ledger.proof_strength

    jd_set = {s.strip().title() for s in required_skills}
    relevant = [e for e in ledger.entries if e.skill in jd_set]

    if not relevant:
        return 0.0

    weights = {
        VerificationStatus.VERIFIED:    1.0,
        VerificationStatus.WEAK:        0.6,
        VerificationStatus.INFERRED:    0.3,
        VerificationStatus.UNSUPPORTED: 0.0,
        VerificationStatus.PENDING:     0.1,
    }
    score = sum(weights.get(e.verification_status, 0.0) for e in relevant)
    return round(score / len(relevant), 4)


def _red_flags(ledger: CandidateLedger, required_skills: list[str]) -> list[str]:
    """Identify critical unsupported skills."""
    required_set = {s.strip().title() for s in required_skills}
    flags: list[str] = []

    # Required skills with zero evidence
    unsupported_required = [
        e.skill for e in ledger.entries
        if e.skill in required_set and e.is_unsupported
    ]
    if unsupported_required:
        flags.append(
            f"Required skills with no evidence: {', '.join(unsupported_required[:4])}"
        )

    # High unsupported ratio overall
    if ledger.total_claims > 0:
        unsup_ratio = ledger.unsupported_count / ledger.total_claims
        if unsup_ratio > 0.5:
            flags.append(
                f"{unsup_ratio:.0%} of all claims have no supporting evidence"
            )

    return flags


def _evaluate_no_ledger(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
) -> AgentVerdict:
    """Fallback when no evidence ledger is available."""
    proxy = min(1.0, (profile.production_signal + profile.achievement_signal) / 2)

    flags: list[str] = ["No evidence ledger available — scoring from resume signals only"]
    if proxy >= 0.6:
        flags.append("Strong production and achievement signals in resume")
    elif proxy <= 0.2:
        flags.append("Weak production evidence in resume")

    return AgentVerdict(
        agent=AgentRole.VERIFICATION,
        candidate_id=profile.candidate_id,
        score=proxy,
        reasoning=(
            f"No evidence ledger available. "
            f"Proxy score based on production signal ({profile.production_signal:.2f}) "
            f"and achievement signal ({profile.achievement_signal:.2f})."
        ),
        flags=flags,
        evidence=[],
    )


def _evaluate_rules(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: CandidateLedger,
) -> AgentVerdict:
    proof    = ledger.proof_strength
    critical = _critical_skill_verification(ledger, hiring_profile.all_required_skill_names)
    flags    = _red_flags(ledger, hiring_profile.all_required_skill_names)

    # Claim density: verified claims relative to total skills claimed
    density = (
        ledger.verified_count / ledger.total_claims
        if ledger.total_claims > 0 else 0.0
    )

    score = round(
        0.40 * proof     +
        0.35 * critical  +
        0.25 * density,
        4,
    )

    if proof >= 0.75:
        flags.append(f"High overall proof strength ({proof:.2f})")
    if critical >= 0.80:
        flags.append("Key required skills are verified")
    elif critical <= 0.20:
        flags.append("Required skills mostly unverified")

    reasoning = (
        f"Overall proof strength: {proof:.2f} "
        f"({ledger.verified_count} verified, "
        f"{ledger.weak_count} weak, "
        f"{ledger.unsupported_count} unsupported out of {ledger.total_claims} claims). "
        f"Critical skill verification (required JD skills only): {critical:.2f}. "
        f"Verified claim density: {density:.2f}."
    )

    # Best evidence items for the report
    evidence = [
        f"{e.skill}: {e.evidence_snippet[:80]}" if e.evidence_snippet else f"{e.skill} ({e.verification_status.value})"
        for e in ledger.entries
        if e.is_verified
    ][:5]

    return AgentVerdict(
        agent=AgentRole.VERIFICATION,
        candidate_id=profile.candidate_id,
        score=min(1.0, score),
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
    ledger: CandidateLedger,
) -> AgentVerdict:
    from ..llm_provider import chat_completion
    import json

    verified_list   = ", ".join(ledger.verified_skills[:8]) or "none"
    unsupported_list = ", ".join(ledger.unsupported_skills[:6]) or "none"

    system_prompt = (
        "You are a Verification Agent. Evaluate how well the candidate's claims are "
        "backed by evidence. Return ONLY JSON with keys: score (float 0-1), "
        "reasoning (string, cite specific verified/unverified claims), "
        "flags (list), evidence (list of URL or snippet strings)."
    )
    user_prompt = f"""Job: {hiring_profile.job_title}
Required skills: {', '.join(hiring_profile.all_required_skill_names)}

Candidate: {profile.name or profile.candidate_id}
Total claims: {ledger.total_claims}
Verified: {ledger.verified_count} | Weak: {ledger.weak_count} | Unsupported: {ledger.unsupported_count}
Proof strength: {ledger.proof_strength:.2f}

Verified skills: {verified_list}
Unsupported skills: {unsupported_list}

Evaluate evidence quality. Return JSON."""

    raw = chat_completion(system_prompt, user_prompt)
    try:
        data = json.loads(raw.strip().strip("```json").strip("```"))
        return AgentVerdict(
            agent=AgentRole.VERIFICATION,
            candidate_id=profile.candidate_id,
            score=float(data.get("score", 0.5)),
            reasoning=str(data.get("reasoning", "")),
            flags=list(data.get("flags", [])),
            evidence=list(data.get("evidence", [])),
        )
    except Exception as exc:
        logger.warning("Verification LLM parse failed (%s) — using rules.", exc)
        return _evaluate_rules(profile, hiring_profile, ledger)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_verification(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: Optional[CandidateLedger] = None,
    *,
    force_fallback: bool = False,
) -> AgentVerdict:
    """
    Run the Verification Agent for one candidate.

    Args:
        profile:        CandidateProfile from Layer 4.
        hiring_profile: HiringProfile from Layer 2.
        ledger:         CandidateLedger from Layer 6.
                        If None, falls back to resume signal proxy.
        force_fallback: Always use rule-based evaluation.

    Returns:
        AgentVerdict with proof_strength and claim verification score.
    """
    if ledger is None:
        return _evaluate_no_ledger(profile, hiring_profile)

    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))
    if force_fallback or not has_key:
        return _evaluate_rules(profile, hiring_profile, ledger)
    try:
        return _evaluate_llm(profile, hiring_profile, ledger)
    except Exception as exc:
        logger.warning("Verification LLM failed (%s) — using rules.", exc)
        return _evaluate_rules(profile, hiring_profile, ledger)
