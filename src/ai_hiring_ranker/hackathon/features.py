"""Offline feature extraction for hackathon precompute cache."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ..agents.orchestrator import evaluate_candidate
from ..candidate_extraction.extractor import extract_candidate_profile
from ..candidate_extraction.schemas import SkillClaim, SkillConfidence, CareerRole, EmploymentCategory
from ..ingestion.schemas import CandidateInput
from ..jd_intelligence.schemas import HiringProfile
from ..retrieval.graph_retriever import get_matched_skills
from ..scoring.scorer import get_scoring_weights, score_one
from .dataset import (
    candidate_id as cid_from_record,
    days_since_active,
    github_activity_score,
    job_title,
    recruiter_response_rate,
    resume_text,
    skills_list,
    years_experience,
)
from ..candidate_extraction.extractor import (
    _compute_career_growth_signal,
    _compute_leadership_signal,
    _compute_production_signal,
    _compute_achievement_signal,
    _sentences,
    _extract_achievements,
)

from .evidence import build_hackathon_ledger
from .guards import honeypot_risk, keyword_stuffer_risk


logger = logging.getLogger(__name__)


def _keyword_match_score(profile_text: str, hiring_profile: HiringProfile) -> float:
    """Lightweight BM25-style keyword overlap (single candidate, no corpus index)."""
    if not profile_text:
        return 0.0
    lowered = profile_text.lower()
    terms = (
        hiring_profile.all_required_skill_names
        + hiring_profile.all_preferred_skill_names
        + hiring_profile.key_responsibilities[:5]
    )
    if not terms:
        return 0.0

    hits = 0.0
    for term in terms:
        token = term.lower().strip()
        if not token:
            continue
        if token in lowered:
            hits += 1.0
            continue
        for piece in re.findall(r"[a-z0-9+#/.-]+", token):
            if len(piece) > 2 and piece in lowered:
                hits += 0.5
                break

    return round(min(1.0, hits / max(len(terms), 1)), 4)


def engagement_multiplier(
    days_inactive: Optional[int],
    response_rate: Optional[float],
) -> float:
    mult = 1.0
    if days_inactive is not None:
        if days_inactive > 730:
            mult *= 0.65
        elif days_inactive > 365:
            mult *= 0.80
        elif days_inactive > 180:
            mult *= 0.92
    if response_rate is not None:
        if response_rate < 0.05:
            mult *= 0.60
        elif response_rate < 0.20:
            mult *= 0.78
        elif response_rate < 0.40:
            mult *= 0.90
    return round(mult, 4)


_TRANSITION_PHRASES = [
    "transitioning toward", "transitioning to", "not the core of my day",
    "not the core of my role", "interested in transitioning",
    "looking to move into", "hoping to move into",
    "while learning modern", "build competence on the",
]


def aspiring_transition_multiplier(record: dict, hiring_profile: HiringProfile) -> float:
    """
    Down-weight candidates who self-disclose that the JD's core domain is NOT
    their current day-to-day work — mirrors the keyword-stuffer trap, but for
    candidates whose skills list overlaps while their own words and current
    title say the domain isn't their core competency yet.
    """
    profile = record.get("profile", {}) or {}
    summary = (profile.get("summary") or "").lower()
    current_title = (profile.get("current_title") or "").lower()

    job_title_lower = (hiring_profile.job_title or "").lower()
    jd_words = {
        w for w in job_title_lower.split()
        if len(w) > 3 and w not in {"senior", "junior", "engineer", "manager"}
    }
    title_overlaps = any(w in current_title for w in jd_words)
    has_hedge = any(phrase in summary for phrase in _TRANSITION_PHRASES)

    if has_hedge and not title_overlaps:
        return 0.80
    return 1.0


def build_features(
    record: dict[str, Any],
    hiring_profile: HiringProfile,
    *,
    line_no: int,
    force_fallback: bool = True,
) -> dict[str, Any]:
    """
    Build compact offline features for one candidate using V2 layers 4–10.

    No GitHub API calls — uses ``github_activity_score`` from the dataset only.
    """
    cid = cid_from_record(record, line_no)
    text = resume_text(record)
    listed_skills = skills_list(record)
    title = job_title(record)
    years = years_experience(record)
    gh_score = github_activity_score(record)
    inactive_days = days_since_active(record)
    response_rate = recruiter_response_rate(record)

    candidate_input = CandidateInput(
        candidate_id=cid,
        raw_text=text or f"Skills: {', '.join(listed_skills)}. Title: {title}.",
    )
    profile = extract_candidate_profile(candidate_input, force_fallback=force_fallback)

    # ── Override regex-guessed fields with real structured data ──────────────
    career_history = record.get("career_history", [])
    real_timeline: list[CareerRole] = []
    for role in career_history:
        duration_months = role.get("duration_months")
        real_timeline.append(CareerRole(
            title=role.get("title", "") or "",
            company=role.get("company", "") or "",
            start_year=int(role["start_date"][:4]) if role.get("start_date") else None,
            end_year=int(role["end_date"][:4]) if role.get("end_date") else None,
            duration_years=round(duration_months / 12, 2) if duration_months else None,
            category=EmploymentCategory.FULL_TIME,
        ))

    description_text = " ".join(
        role.get("description", "") or "" for role in career_history
    )
    anonymized_name = (record.get("profile", {}) or {}).get("anonymized_name")
    if anonymized_name:
        profile.name = str(anonymized_name).strip()

    if real_timeline:
        profile.career_timeline = real_timeline
        profile.career_growth_signal = _compute_career_growth_signal(real_timeline)
    if years is not None:
        profile.total_years_experience = years
    if description_text.strip():
        profile.leadership_signal = _compute_leadership_signal(description_text)
        profile.production_signal = _compute_production_signal(_sentences(description_text))
        profile.achievement_signal = _compute_achievement_signal(
            _extract_achievements(_sentences(description_text))
        )

    if listed_skills:
        existing = {s.skill.lower() for s in profile.skills}
        for skill in listed_skills:
            norm = skill.strip().title()
            if norm.lower() not in existing:
                profile.skills.append(
                    SkillClaim(skill=norm, confidence=SkillConfidence.EXPLICIT, evidence_snippets=[])
                )

    ledger = build_hackathon_ledger(
        profile,
        github_activity_score=gh_score,
        run_id="precompute",
    )
    eval_result = evaluate_candidate(
        profile,
        hiring_profile,
        ledger,
        force_fallback=True,
    )
    candidate_score = score_one(
        eval_result,
        get_scoring_weights(),
        hiring_profile,
        ledger,
    )

    matched_req, matched_pref, graph_expanded = get_matched_skills(profile, hiring_profile)
    keyword_score = _keyword_match_score(text, hiring_profile)

    honeypot, honeypot_flags = honeypot_risk(record)
    stuffer, stuffer_flags = keyword_stuffer_risk(record, hiring_profile.job_title)
    engagement = engagement_multiplier(inactive_days, response_rate)
    transition_mult = aspiring_transition_multiplier(record, hiring_profile)

    dims = {
        "skill_fit": candidate_score.skill_fit,
        "experience_depth": candidate_score.experience_depth,
        "seniority_match": candidate_score.seniority_match,
        "domain_match": candidate_score.domain_match,
        "career_growth": candidate_score.career_growth,
        "proof_strength": candidate_score.proof_strength,
    }

    final_score = float(candidate_score.final_score)
    retrieval_blend = round(0.85 * (final_score / 100.0) + 0.15 * keyword_score, 4)
    final_score = round(retrieval_blend * 100.0, 4)

    if honeypot >= 0.45:
        final_score *= 0.35
    elif honeypot >= 0.25:
        final_score *= 0.60
    if stuffer >= 0.35:
        final_score *= 0.70
    elif stuffer >= 0.20:
        final_score *= 0.85
    final_score *= engagement
    final_score *= transition_mult
    final_score = round(max(0.0, min(100.0, final_score)), 4)

    missing_required = [
        s
        for s in hiring_profile.all_required_skill_names
        if s not in matched_req and s not in graph_expanded
    ]

    strengths = _clean([
        eval_result.strengths[0] if eval_result.strengths else "",
        f"Matches required skills: {', '.join(matched_req[:4])}" if matched_req else "",
        f"Graph-adjacent skills: {', '.join(graph_expanded[:3])}" if graph_expanded else "",
        f"GitHub activity score {gh_score:.2f}" if gh_score >= 0.5 else "",
        (
            f"{years or profile.total_years_experience or 0:.0f} years experience"
            if (years or profile.total_years_experience)
            else ""
        ),
    ])
    risks = _clean([
        eval_result.risks[0] if eval_result.risks else "",
        f"Missing required: {', '.join(missing_required[:3])}" if missing_required else "",
        honeypot_flags[0] if honeypot_flags else "",
        stuffer_flags[0] if stuffer_flags else "",
        "Low platform engagement" if engagement < 0.85 else "",
        "Self-disclosed domain transition — core role not yet ML-focused" if transition_mult < 1.0 else "",
    ])

    return {
        "candidate_id": cid,
        "candidate_name": profile.name or title or cid,
        "job_title": title,
        "years_experience": years if years is not None else profile.total_years_experience,
        "github_activity_score": gh_score,
        "inactive_days": inactive_days,
        "response_rate": response_rate,
        "engagement_multiplier": engagement,
        "transition_multiplier": transition_mult,
        "honeypot_risk": round(honeypot, 4),
        "stuffer_risk": round(stuffer, 4),
        "honeypot_flags": honeypot_flags,
        "stuffer_flags": stuffer_flags,
        "keyword_score": keyword_score,
        "matched_required": matched_req[:12],
        "matched_preferred": matched_pref[:8],
        "partial_matches": graph_expanded[:8],
        "missing_required": missing_required[:8],
        "verified_skills": [s.skill for s in profile.skills if s.confidence.value == "explicit"][:12],
        "strengths": strengths,
        "risks": risks,
        "dimensions": dims,
        "base_score": final_score,
        "final_score": final_score,
        "score_notes": candidate_score.score_notes[:4],
        "agent_summary": eval_result.summary[:240],
    }


def _clean(items: list[str]) -> list[str]:
    return [x.strip() for x in items if x and x.strip()]