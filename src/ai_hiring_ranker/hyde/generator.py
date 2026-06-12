"""
HyDE Generator — Layer 3.

Generates three ideal candidate profiles (Minimum / Strong / Exceptional)
from a HiringProfile produced by Layer 2.

Two execution modes:
  1. LLM mode  — rich, fluent profiles via structured_completion().
                 Requires an API key.  Uses a higher temperature (0.7) than
                 the JD agent because creative variation improves retrieval.
  2. Fallback  — deterministic template-based profiles built from the
                 HiringProfile skills lists.  Less fluent but always works.

Important: HyDE profiles are used ONLY for retrieval (Layer 8).
They are never shown to recruiters or candidates and are never used
directly in scoring.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ..config import LLMConfig, get_llm_config, get_hyde_config
from ..jd_intelligence.schemas import HiringProfile, SeniorityLevel
from ..llm_provider import structured_completion
from .schemas import CandidateTier, HyDEResult, IdealCandidateProfile

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[4] / "prompts" / "hyde_generation.md"





def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a senior recruiter. Generate three hypothetical candidate profiles "
        "(minimum, strong, exceptional) as a JSON object. "
        "Write profile_text in first-person resume style."
    )





def _build_user_prompt(profile: HiringProfile) -> str:
    required = ", ".join(profile.all_required_skill_names) or "not specified"
    preferred = ", ".join(profile.all_preferred_skill_names) or "none"
    responsibilities = "\n".join(f"  - {r}" for r in profile.key_responsibilities[:6])
    hidden = "\n".join(f"  - {h.description}" for h in profile.hidden_expectations[:4])
    years = (
        f"{profile.years_of_experience_min}+" if profile.years_of_experience_min else "not specified"
    )

    return f"""Generate three ideal candidate profiles for this role.

Role: {profile.job_title}
Domain: {profile.domain or "not specified"}
Seniority: {profile.seniority.value}
Years of experience required: {years}

Required skills: {required}
Preferred skills: {preferred}

Key responsibilities:
{responsibilities or "  - not specified"}

Hidden expectations (implied by the JD):
{hidden or "  - none detected"}

Return the JSON object now."""


def _run_llm(profile: HiringProfile) -> HyDEResult:
    hyde_cfg = get_hyde_config()
    llm_cfg = get_llm_config()

    # HyDE uses a higher temperature for richer, more varied profiles
    hyde_llm_cfg = LLMConfig(
        provider=llm_cfg.provider,
        model=llm_cfg.model,
        temperature=hyde_cfg.temperature,   # 0.7 by default
        max_tokens=llm_cfg.max_tokens,
        api_key=llm_cfg.api_key,
    )

    return structured_completion(
        system_prompt=_load_system_prompt(),
        user_prompt=_build_user_prompt(profile),
        schema=HyDEResult,
        config=hyde_llm_cfg,
    )



# ---------------------------------------------------------------------------

# Maps seniority level → (min_years, strong_years, exceptional_years, label_min, label_strong, label_exceptional)
_SENIORITY_YEARS: dict[SeniorityLevel, tuple] = {
    SeniorityLevel.INTERN:     (0, 1, 2,  "0–1 years",  "1–2 years",  "2 years+"),
    SeniorityLevel.JUNIOR:     (1, 2, 3,  "1–2 years",  "2–3 years",  "3 years+"),
    SeniorityLevel.MID:        (3, 4, 5,  "3 years",    "4–5 years",  "5 years+"),
    SeniorityLevel.SENIOR:     (5, 6, 8,  "5 years",    "6–7 years",  "8 years+"),
    SeniorityLevel.LEAD:       (6, 7, 10, "6 years",    "7–8 years",  "10 years+"),
    SeniorityLevel.STAFF:      (7, 8, 12, "7 years",    "8–10 years", "12 years+"),
    SeniorityLevel.PRINCIPAL:  (8, 10,15, "8 years",    "10 years",   "15 years+"),
    SeniorityLevel.MANAGER:    (5, 7, 10, "5 years",    "7 years",    "10 years+"),
    SeniorityLevel.UNKNOWN:    (2, 4, 6,  "2–3 years",  "4–5 years",  "6 years+"),
}


def _fallback_profile_text(
    tier: CandidateTier,
    profile: HiringProfile,
    years: int,
    skills: list[str],
    extra_skills: list[str],
) -> str:
    """Build a simple first-person resume narrative from skill lists."""
    role = profile.job_title
    domain = profile.domain or "technology"
    skill_str = ", ".join(skills[:8]) if skills else "relevant technologies"
    extra_str = (", ".join(extra_skills[:4]) + " and more") if extra_skills else ""

    if tier == CandidateTier.MINIMUM:
        return (
            f"I am a {role} with {years} years of experience in {domain}. "
            f"My core skills include {skill_str}. "
            f"I have completed several projects that demonstrate my ability to apply these skills in practical settings. "
            f"I am comfortable working in a team environment and can contribute to delivering working software. "
            f"I am looking to grow my expertise and take on more responsibility in a professional setting."
        )

    elif tier == CandidateTier.STRONG:
        responsibilities = (
            " and ".join(r.lower()[:60] for r in profile.key_responsibilities[:2])
            if profile.key_responsibilities
            else f"build and deploy {domain} systems"
        )
        return (
            f"I am a {role} with {years} years of hands-on experience in {domain}. "
            f"I have strong proficiency in {skill_str}"
            + (f", as well as {extra_str}" if extra_str else "") + ". "
            f"In my previous roles I have worked to {responsibilities}. "
            f"I have shipped multiple production systems and have a track record of delivering measurable results — "
            f"for example, improving system throughput by over 30% through architectural improvements. "
            f"I take ownership of my work, collaborate well with cross-functional teams, and mentor junior engineers."
        )

    else:  # EXCEPTIONAL
        all_skills = skills + extra_skills
        skill_str_full = ", ".join(all_skills[:10]) if all_skills else skill_str
        return (
            f"I am a {role} with {years}+ years of deep expertise in {domain}. "
            f"I have mastered {skill_str_full}. "
            f"I have led the design and delivery of large-scale {domain} systems serving millions of users, "
            f"achieving significant improvements in performance, reliability, and developer productivity. "
            f"I have a strong track record of technical leadership: defining architecture, driving adoption of best practices, "
            f"and growing engineering teams. "
            f"Beyond my day-to-day work, I have contributed to open-source projects, presented at industry conferences, "
            f"and published technical writing. "
            f"I bring both deep technical depth and the ability to align engineering work with business outcomes."
        )


def _run_fallback(profile: HiringProfile) -> HyDEResult:
    seniority = profile.seniority
    years_tuple = _SENIORITY_YEARS.get(seniority, _SENIORITY_YEARS[SeniorityLevel.UNKNOWN])
    min_yrs, strong_yrs, exc_yrs, lbl_min, lbl_strong, lbl_exc = years_tuple

    required = profile.all_required_skill_names
    preferred = profile.all_preferred_skill_names

    # Override with JD-specified minimum if available
    if profile.years_of_experience_min:
        min_yrs = profile.years_of_experience_min
        strong_yrs = min_yrs + 2
        exc_yrs = min_yrs + 4

    minimum = IdealCandidateProfile(
        tier=CandidateTier.MINIMUM,
        profile_text=_fallback_profile_text(CandidateTier.MINIMUM, profile, min_yrs, required, []),
        skills_demonstrated=required,
        experience_years=min_yrs,
        seniority_label=lbl_min,
        differentiator="Meets the minimum bar — all required skills, no extras.",
    )

    strong = IdealCandidateProfile(
        tier=CandidateTier.STRONG,
        profile_text=_fallback_profile_text(CandidateTier.STRONG, profile, strong_yrs, required, preferred),
        skills_demonstrated=required + preferred,
        experience_years=strong_yrs,
        seniority_label=lbl_strong,
        differentiator="Clearly qualified — required + preferred skills with production evidence.",
    )

    exceptional = IdealCandidateProfile(
        tier=CandidateTier.EXCEPTIONAL,
        profile_text=_fallback_profile_text(CandidateTier.EXCEPTIONAL, profile, exc_yrs, required, preferred),
        skills_demonstrated=required + preferred,
        experience_years=exc_yrs,
        seniority_label=lbl_exc,
        differentiator="Exceptional — deep expertise, leadership, and measurable impact at scale.",
    )

    return HyDEResult(
        job_title=profile.job_title,
        domain=profile.domain,
        profiles=[minimum, strong, exceptional],
    )




def generate_hyde_profiles(
    profile: HiringProfile,
    *,
    force_fallback: bool = False,
) -> HyDEResult:
    """
    Generate three ideal candidate profiles from a HiringProfile.

    Automatically uses LLM mode if an API key is available, otherwise
    falls back to template-based generation.

    Args:
        profile:        Validated HiringProfile from Layer 2.
        force_fallback: Always use template mode (for testing / offline use).

    Returns:
        HyDEResult with minimum, strong, and exceptional profiles.
    """
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))

    if force_fallback or not has_key:
        mode = "template fallback (no API key)" if not has_key else "template fallback (forced)"
        logger.info("HyDE Generator running in %s mode", mode)
        return _run_fallback(profile)

    logger.info("HyDE Generator running in LLM mode (temperature=%.1f)", get_hyde_config().temperature)
    try:
        return _run_llm(profile)
    except Exception as exc:
        logger.warning("LLM mode failed (%s) — falling back to template generation.", exc)
        return _run_fallback(profile)
