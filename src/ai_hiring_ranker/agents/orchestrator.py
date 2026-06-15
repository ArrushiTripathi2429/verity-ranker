"""
Multi-Agent Evaluation Orchestrator — Layer 9.

Runs all four specialist agents for every shortlisted candidate, then
synthesises their verdicts into a final EvaluationResult with:
  - Six dimension scores (feeds Layer 10 Rubric Scoring)
  - Strengths + risks (feeds Layer 12 Recruiter Report)
  - A 2–3 sentence overall summary

Agent execution order:
  1. JD Fit Agent          → skill / requirement coverage
  2. Technical Fit Agent   → depth, production evidence, verification
  3. Career Trajectory Agent → seniority, experience, growth
  4. Verification Agent    → evidence ledger proof strength
  5. Final Synthesis       → combines all four into DimensionScores

Agents run independently (no inter-agent dependency during scoring).
The synthesis step is the only point where all four verdicts are visible
together.

Public API
----------
evaluate_candidate(profile, hiring_profile, ledger, force_fallback)
    → EvaluationResult

evaluate_all(profiles, hiring_profile, ledger_map, shortlist_ids, force_fallback)
    → BatchEvaluationResult
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile
from ..evidence.schemas import CandidateLedger
from ..jd_intelligence.schemas import HiringProfile
from .jd_fit_agent import evaluate_jd_fit
from .schemas import (
    AgentRole,
    AgentVerdict,
    BatchEvaluationResult,
    DimensionScores,
    EvaluationResult,
)
from .technical_fit_agent import evaluate_technical_fit
from .trajectory_agent import evaluate_trajectory
from .verification_agent import evaluate_verification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------


def _synthesise_dimensions(
    jd_verdict:   AgentVerdict,
    tech_verdict: AgentVerdict,
    traj_verdict: AgentVerdict,
    verif_verdict: AgentVerdict,
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
) -> DimensionScores:
    """
    Map the four agent scores to the six rubric dimensions.

    Mapping rationale:
      skill_fit        ← JD Fit (primary) + Technical breadth (secondary)
      experience_depth ← Trajectory experience sub-score
      seniority_match  ← Trajectory seniority sub-score
      domain_match     ← JD Fit (domain alignment sub-score)
      career_growth    ← Trajectory growth sub-score
      proof_strength   ← Verification Agent score
    """
    # Skill fit: blend of JD fit + technical fit
    skill_fit = round(0.65 * jd_verdict.score + 0.35 * tech_verdict.score, 4)

    # Experience depth: trajectory agent carries this but we cross-check
    # with the overall years signal from the profile
    from .trajectory_agent import _experience_depth_score, _seniority_match_score, _career_growth_score
    exp_depth  = round(_experience_depth_score(profile, hiring_profile), 4)
    sen_match  = round(_seniority_match_score(profile, hiring_profile), 4)
    growth     = round(_career_growth_score(profile), 4)

    # Domain match: inferred from JD fit reasoning and profile domain signals
    # Use the JD fit score with a small weight toward the trajectory score
    domain_match = round(
        0.70 * jd_verdict.score + 0.30 * traj_verdict.score,
        4,
    )
    # Clamp domain to realistic range — it shouldn't fully track JD fit
    domain_match = round(min(domain_match, 0.95), 4)

    return DimensionScores(
        skill_fit=min(1.0, skill_fit),
        experience_depth=min(1.0, exp_depth),
        seniority_match=min(1.0, sen_match),
        domain_match=min(1.0, domain_match),
        career_growth=min(1.0, growth),
        proof_strength=min(1.0, verif_verdict.score),
    )


def _collect_strengths_and_risks(
    verdicts: list[AgentVerdict],
    profile: CandidateProfile,
    dimensions: DimensionScores,
) -> tuple[list[str], list[str]]:
    """Extract top strengths and risks from all agent flags."""
    strengths: list[str] = []
    risks:     list[str] = []

    # Aggregate flags from all agents
    for verdict in verdicts:
        for flag in verdict.flags:
            flag_lower = flag.lower()
            is_negative = any(
                kw in flag_lower for kw in
                ["missing", "gap", "low", "weak", "mismatch", "under", "few",
                 "no evidence", "unverified", "unsupported", "fraction"]
            )
            if is_negative:
                risks.append(flag)
            else:
                strengths.append(flag)

    # Dimension-driven strengths
    if dimensions.skill_fit >= 0.80:
        strengths.insert(0, f"Strong JD skill alignment ({dimensions.skill_fit:.0%} fit)")
    if dimensions.proof_strength >= 0.75:
        strengths.insert(0, "Claims are well-verified by external evidence")
    if profile.achievement_signal >= 0.65:
        strengths.append("Demonstrates measurable achievements")

    # Dimension-driven risks
    if dimensions.skill_fit < 0.40:
        risks.insert(0, f"Low JD skill fit ({dimensions.skill_fit:.0%})")
    if dimensions.experience_depth < 0.40:
        risks.append("Under the minimum experience requirement")
    if dimensions.proof_strength < 0.25:
        risks.append("Most claims are unverified — treat profile at face value")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_strengths: list[str] = []
    for s in strengths:
        if s not in seen:
            seen.add(s)
            unique_strengths.append(s)

    seen = set()
    unique_risks: list[str] = []
    for r in risks:
        if r not in seen:
            seen.add(r)
            unique_risks.append(r)

    return unique_strengths[:5], unique_risks[:3]


def _build_summary(
    profile: CandidateProfile,
    dimensions: DimensionScores,
    strengths: list[str],
    risks: list[str],
) -> str:
    overall = round(
        (dimensions.skill_fit + dimensions.experience_depth +
         dimensions.seniority_match + dimensions.domain_match +
         dimensions.career_growth + dimensions.proof_strength) / 6,
        2,
    )
    label = "strong" if overall >= 0.70 else "moderate" if overall >= 0.45 else "weak"
    top_strength = strengths[0] if strengths else "relevant background"
    top_risk = risks[0] if risks else "no critical concerns"

    return (
        f"{profile.name or profile.candidate_id} is a {label} candidate overall "
        f"(score: {overall:.2f}). "
        f"Key strength: {top_strength}. "
        f"Primary concern: {top_risk}."
    )


# ---------------------------------------------------------------------------
# LLM synthesis (optional enrichment)
# ---------------------------------------------------------------------------


def _llm_synthesise(
    profile: CandidateProfile,
    verdicts: list[AgentVerdict],
    dimensions: DimensionScores,
    hiring_profile: HiringProfile,
) -> tuple[list[str], list[str], str]:
    """Use LLM to write richer strengths, risks, and summary."""
    from ..llm_provider import chat_completion
    import json

    verdicts_text = "\n".join(
        f"  {v.agent.value}: score={v.score:.2f}  {v.reasoning[:120]}"
        for v in verdicts
    )
    dim_text = (
        f"skill_fit={dimensions.skill_fit:.2f}  exp={dimensions.experience_depth:.2f}  "
        f"seniority={dimensions.seniority_match:.2f}  domain={dimensions.domain_match:.2f}  "
        f"growth={dimensions.career_growth:.2f}  proof={dimensions.proof_strength:.2f}"
    )

    system_prompt = (
        "You are a Final Ranking Agent. Synthesise the verdicts from all specialist "
        "agents into a concise evaluation. Return ONLY JSON with keys: "
        "strengths (list of 3–5 short strings, evidence-cited), "
        "risks (list of 1–3 short strings), "
        "summary (string, 2–3 sentences)."
    )
    user_prompt = f"""Job: {hiring_profile.job_title} ({hiring_profile.seniority.value})
Candidate: {profile.name or profile.candidate_id}

Agent verdicts:
{verdicts_text}

Dimension scores: {dim_text}

Write the final synthesis. Be concise and cite evidence."""

    raw = chat_completion(system_prompt, user_prompt)
    data = json.loads(raw.strip().strip("```json").strip("```"))
    return (
        list(data.get("strengths", [])),
        list(data.get("risks", [])),
        str(data.get("summary", "")),
    )


# ---------------------------------------------------------------------------
# Per-candidate evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate(
    profile: CandidateProfile,
    hiring_profile: HiringProfile,
    ledger: Optional[CandidateLedger] = None,
    *,
    force_fallback: bool = False,
) -> EvaluationResult:
    """
    Run all four specialist agents + synthesis for one candidate.

    Args:
        profile:        CandidateProfile from Layer 4.
        hiring_profile: HiringProfile from Layer 2.
        ledger:         CandidateLedger from Layer 6 (optional but recommended).
        force_fallback: Use rule-based agents only (no LLM calls).

    Returns:
        EvaluationResult with six dimension scores, strengths, risks, summary.
    """
    logger.info("Evaluating candidate: %s", profile.candidate_id)

    # ── Run all four agents ─────────────────────────────────────────────
    jd_verdict    = evaluate_jd_fit(profile, hiring_profile, ledger, force_fallback=force_fallback)
    tech_verdict  = evaluate_technical_fit(profile, hiring_profile, ledger, force_fallback=force_fallback)
    traj_verdict  = evaluate_trajectory(profile, hiring_profile, ledger, force_fallback=force_fallback)
    verif_verdict = evaluate_verification(profile, hiring_profile, ledger, force_fallback=force_fallback)

    verdicts = [jd_verdict, tech_verdict, traj_verdict, verif_verdict]

    # ── Synthesise dimension scores ─────────────────────────────────────
    dimensions = _synthesise_dimensions(
        jd_verdict, tech_verdict, traj_verdict, verif_verdict,
        profile, hiring_profile,
    )

    # ── Collect strengths / risks ───────────────────────────────────────
    strengths, risks = _collect_strengths_and_risks(verdicts, profile, dimensions)

    # ── Build summary ───────────────────────────────────────────────────
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))
    summary = ""

    if not force_fallback and has_key:
        try:
            llm_strengths, llm_risks, llm_summary = _llm_synthesise(
                profile, verdicts, dimensions, hiring_profile
            )
            if llm_strengths:
                strengths = llm_strengths
            if llm_risks:
                risks = llm_risks
            if llm_summary:
                summary = llm_summary
        except Exception as exc:
            logger.warning(
                "LLM synthesis failed for %s (%s) — using rule-based summary.",
                profile.candidate_id, exc,
            )

    if not summary:
        summary = _build_summary(profile, dimensions, strengths, risks)

    result = EvaluationResult(
        candidate_id=profile.candidate_id,
        candidate_name=profile.name,
        verdicts=verdicts,
        dimensions=dimensions,
        strengths=strengths,
        risks=risks,
        summary=summary,
    )

    logger.info(
        "  %s  overall=%.3f  skl=%.2f exp=%.2f sen=%.2f dom=%.2f trj=%.2f prf=%.2f",
        profile.candidate_id,
        result.overall_score,
        dimensions.skill_fit,
        dimensions.experience_depth,
        dimensions.seniority_match,
        dimensions.domain_match,
        dimensions.career_growth,
        dimensions.proof_strength,
    )

    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_all(
    profiles: list[CandidateProfile],
    hiring_profile: HiringProfile,
    ledger_map: Optional[dict[str, CandidateLedger]] = None,
    *,
    shortlist_ids: Optional[list[str]] = None,
    force_fallback: bool = False,
) -> BatchEvaluationResult:
    """
    Evaluate all shortlisted candidates.

    Args:
        profiles:       All CandidateProfiles from Layer 4.
        hiring_profile: HiringProfile from Layer 2.
        ledger_map:     dict[candidate_id → CandidateLedger] from Layer 6.
        shortlist_ids:  Only evaluate these candidates (from Layer 8 shortlist).
                        If None, evaluates all profiles.
        force_fallback: Use rule-based agents only.

    Returns:
        BatchEvaluationResult sorted by overall_score descending.
    """
    # Filter to shortlist if provided
    if shortlist_ids is not None:
        shortlist_set = set(shortlist_ids)
        profiles = [p for p in profiles if p.candidate_id in shortlist_set]

    logger.info(
        "Layer 9: evaluating %d candidates (force_fallback=%s)",
        len(profiles),
        force_fallback,
    )

    results: list[EvaluationResult] = []
    for profile in profiles:
        ledger = (ledger_map or {}).get(profile.candidate_id)
        try:
            result = evaluate_candidate(
                profile, hiring_profile, ledger,
                force_fallback=force_fallback,
            )
        except Exception as exc:
            logger.error(
                "Evaluation crashed for %s: %s — inserting empty result.",
                profile.candidate_id, exc,
            )
            result = EvaluationResult(
                candidate_id=profile.candidate_id,
                candidate_name=profile.name,
                summary=f"Evaluation failed: {exc}",
            )
        results.append(result)

    batch = BatchEvaluationResult(
        job_title=hiring_profile.job_title,
        results=results,
    )

    logger.info(
        "Layer 9 complete: %d candidates evaluated.\n%s",
        len(results),
        batch.summary_table(),
    )
    return batch
