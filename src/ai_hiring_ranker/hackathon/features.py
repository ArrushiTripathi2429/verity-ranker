"""Offline feature extraction for hackathon precompute cache."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..agents.jd_fit_agent import _skill_coverage
from ..agents.trajectory_agent import (
    _career_growth_score,
    _experience_depth_score,
    _seniority_match_score,
)
from ..candidate_extraction.extractor import extract_candidate_profile
from ..ingestion.schemas import CandidateInput
from ..jd_intelligence.schemas import HiringProfile
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
from .guards import honeypot_risk, keyword_stuffer_risk

logger = logging.getLogger(__name__)


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


def build_features(
    record: dict[str, Any],
    hiring_profile: HiringProfile,
    *,
    line_no: int,
    force_fallback: bool = True,
) -> dict[str, Any]:
    """
    Build compact offline features for one candidate.

    No GitHub API calls — uses github_activity_score from the dataset only.
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

    if listed_skills and len(profile.skills) < len(listed_skills):
        from ..candidate_extraction.schemas import SkillClaim, SkillConfidence

        existing = {s.skill.lower() for s in profile.skills}
        for skill in listed_skills:
            norm = skill.strip().title()
            if norm.lower() not in existing:
                profile.skills.append(
                    SkillClaim(skill=norm, confidence=SkillConfidence.EXPLICIT, evidence_snippets=[])
                )

    req_cov, req_exact, req_partial = _skill_coverage(
        profile.skill_names,
        hiring_profile.all_required_skill_names,
    )
    pref_cov, pref_exact, pref_partial = _skill_coverage(
        profile.skill_names,
        hiring_profile.all_preferred_skill_names,
        min_weight=0.50,
    )

    skill_fit = round(0.70 * req_cov + 0.30 * pref_cov, 4)
    experience_depth = round(_experience_depth_score(profile, hiring_profile), 4)
    seniority_match = round(_seniority_match_score(profile, hiring_profile), 4)
    career_growth = round(_career_growth_score(profile), 4)
    proof_strength = round(0.55 * gh_score + 0.45 * min(1.0, profile.production_signal + 0.15), 4)

    domain_match = 0.5
    if hiring_profile.domain:
        blob = " ".join(
            [title]
            + [r.title for r in profile.career_timeline]
            + [p.description for p in profile.projects]
        ).lower()
        domain_match = 0.8 if hiring_profile.domain.lower() in blob else 0.35

    honeypot, honeypot_flags = honeypot_risk(record)
    stuffer, stuffer_flags = keyword_stuffer_risk(record, hiring_profile.job_title)
    engagement = engagement_multiplier(inactive_days, response_rate)

    weights = {
        "skill_fit": 0.30,
        "experience_depth": 0.20,
        "seniority_match": 0.15,
        "domain_match": 0.15,
        "career_growth": 0.10,
        "proof_strength": 0.10,
    }
    dims = {
        "skill_fit": skill_fit,
        "experience_depth": experience_depth,
        "seniority_match": seniority_match,
        "domain_match": domain_match,
        "career_growth": career_growth,
        "proof_strength": proof_strength,
    }
    base = sum(dims[k] * weights[k] for k in weights) * 100.0

    if honeypot >= 0.45:
        base *= 0.35
    elif honeypot >= 0.25:
        base *= 0.60
    if stuffer >= 0.35:
        base *= 0.70
    elif stuffer >= 0.20:
        base *= 0.85
    base *= engagement

    missing_required = [
        s for s in hiring_profile.all_required_skill_names
        if s not in req_exact and s not in req_partial
    ]

    return {
        "candidate_id": cid,
        "candidate_name": profile.name or title or cid,
        "job_title": title,
        "years_experience": years if years is not None else profile.total_years_experience,
        "github_activity_score": gh_score,
        "inactive_days": inactive_days,
        "response_rate": response_rate,
        "engagement_multiplier": engagement,
        "honeypot_risk": round(honeypot, 4),
        "stuffer_risk": round(stuffer, 4),
        "honeypot_flags": honeypot_flags,
        "stuffer_flags": stuffer_flags,
        "matched_required": req_exact[:12],
        "matched_preferred": pref_exact[:8],
        "partial_matches": req_partial[:8],
        "missing_required": missing_required[:8],
        "verified_skills": [
            s.skill for s in profile.skills if s.confidence.value == "explicit"
        ][:12],
        "strengths": [
            f"Matches required skills: {', '.join(req_exact[:4])}" if req_exact else "",
            f"GitHub activity score {gh_score:.2f}" if gh_score >= 0.5 else "",
            (
                f"{years or profile.total_years_experience or 0:.0f} years experience"
                if (years or profile.total_years_experience)
                else ""
            ),
        ],
        "risks": [
            f"Missing required: {', '.join(missing_required[:3])}" if missing_required else "",
            honeypot_flags[0] if honeypot_flags else "",
            stuffer_flags[0] if stuffer_flags else "",
            "Low platform engagement" if engagement < 0.85 else "",
        ],
        "dimensions": dims,
        "base_score": round(max(0.0, min(100.0, base)), 4),
    }
