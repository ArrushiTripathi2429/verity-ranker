"""Offline feature extraction for hackathon precompute cache."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
from datetime import datetime

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

# ─── JD LOCATION PREFERENCES (from spec) ────────────────────────────────────
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
    "delhi ncr", "hyderabad", "mumbai", "bangalore",
}
TIER_1_CITIES = {
    "delhi", "mumbai", "bangalore", "hyderabad", "pune", "ncr",
    "gurugram", "gurgaon", "noida", "chandigarh",
}

# ─── JD DISQUALIFIER RULES ──────────────────────────────────────────────────
CONSULTING_TRAP_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "deloitte", "pwc", "kpmg", "ibm consulting", "mckinsey",
    "bain", "bcg", "goldman sachs", "jpmorgan",  # Extended list
}

PURE_RESEARCH_SIGNALS = {
    "phd", "postdoc", "research scientist", "academic", "university",
    "published", "paper", "journal", "conference", "arxiv",
}


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


# ─── NEW BEHAVIORAL SIGNAL EXTRACTORS ────────────────────────────────────────

def _get_open_to_work_flag(record: dict) -> bool:
    """Extract open_to_work_flag from profile."""
    profile = record.get("profile") or {}
    return bool(profile.get("open_to_work_flag", False))


def _get_notice_period_days(record: dict) -> Optional[int]:
    """Extract notice_period_days from profile."""
    profile = record.get("profile") or {}
    try:
        val = profile.get("notice_period_days")
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_preferred_work_mode(record: dict) -> Optional[str]:
    """Extract preferred_work_mode (remote, hybrid, on-site)."""
    profile = record.get("profile") or {}
    return profile.get("preferred_work_mode", "").lower()


def _get_willing_to_relocate(record: dict) -> bool:
    """Extract willing_to_relocate flag."""
    profile = record.get("profile") or {}
    return bool(profile.get("willing_to_relocate", False))


def _get_expected_salary_inr_lpa(record: dict) -> Optional[float]:
    """Extract expected_salary_range_inr_lpa (in Lakhs per annum)."""
    profile = record.get("profile") or {}
    try:
        val = profile.get("expected_salary_range_inr_lpa")
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_avg_response_time_hours(record: dict) -> Optional[float]:
    """Extract avg_response_time_hours from profile."""
    profile = record.get("profile") or {}
    try:
        val = profile.get("avg_response_time_hours")
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_applications_submitted_30d(record: dict) -> Optional[int]:
    """Extract applications_submitted_30d."""
    profile = record.get("profile") or {}
    try:
        val = profile.get("applications_submitted_30d")
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_profile_views_received_30d(record: dict) -> Optional[int]:
    """Extract profile_views_received_30d."""
    profile = record.get("profile") or {}
    try:
        val = profile.get("profile_views_received_30d")
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_saved_by_recruiters_30d(record: dict) -> Optional[int]:
    """Extract saved_by_recruiters_30d."""
    profile = record.get("profile") or {}
    try:
        val = profile.get("saved_by_recruiters_30d")
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_offer_acceptance_rate(record: dict) -> Optional[float]:
    """
    Extract offer_acceptance_rate.
    -1 means no prior offers; low (0.0-0.3) suggests flight risk; high (0.8+) is good.
    """
    profile = record.get("profile") or {}
    try:
        val = profile.get("offer_acceptance_rate")
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_connection_count(record: dict) -> Optional[int]:
    """Extract connection_count (network size signal)."""
    profile = record.get("profile") or {}
    try:
        val = profile.get("connection_count")
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_endorsements_received(record: dict) -> Optional[int]:
    """Extract endorsements_received (peer validation signal)."""
    profile = record.get("profile") or {}
    try:
        val = profile.get("endorsements_received")
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_location_info(record: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract (location_city, country) from profile."""
    profile = record.get("profile") or {}
    location = profile.get("location", "").lower().strip()
    country = profile.get("country", "").lower().strip()
    return (location if location else None, country if country else None)


def _get_education_tier(record: dict) -> Optional[str]:
    """Extract education tier from education array (tier_1 through tier_4, unknown)."""
    education = record.get("education") or []
    if isinstance(education, list) and education:
        return education[0].get("tier", "").lower()
    return None


# ─── BEHAVIORAL SIGNAL MULTIPLIERS ──────────────────────────────────────────

def open_to_work_multiplier(is_open: bool) -> float:
    """JD: Candidates not open to work should be heavily down-weighted."""
    return 1.0 if is_open else 0.5


def notice_period_multiplier(notice_days: Optional[int]) -> float:
    """JD: 'We'd love sub-30-day notice. 30+ day candidates are still in scope but bar gets higher.'"""
    if notice_days is None:
        return 0.95  # Unknown = slight penalty
    if notice_days <= 14:
        return 1.0
    elif notice_days <= 30:
        return 0.95
    elif notice_days <= 60:
        return 0.85
    else:
        return 0.70


def work_mode_multiplier(preferred_mode: Optional[str], jd_requires_hybrid: bool = True) -> float:
    """JD says 'Pune/Noida (Hybrid)' — remote-only is worse fit."""
    if not preferred_mode or preferred_mode in ("hybrid", "on-site", "on site"):
        return 1.0
    if preferred_mode == "remote":
        return 0.75  # Remote-only is less preferred but not disqualifying
    return 0.95


def relocation_multiplier(willing: bool, location: Optional[str]) -> float:
    """
    JD: 'Open to relocation candidates from Tier-1 Indian cities.'
    Already in Tier-1 is best; willing to relocate is good; unwilling outside Tier-1 is bad.
    """
    if not location:
        return 0.90  # Unknown location = slight penalty
    
    location_lower = location.lower()
    in_preferred = any(city in location_lower for city in PREFERRED_LOCATIONS)
    in_tier1 = any(city in location_lower for city in TIER_1_CITIES)
    
    if in_preferred:
        return 1.0
    elif in_tier1:
        return 0.95
    elif willing:
        return 0.85  # Outside Tier-1 but willing to relocate
    else:
        return 0.60  # Outside Tier-1 and unwilling


def response_time_multiplier(avg_response_hours: Optional[float]) -> float:
    """Slower responders are harder to hire."""
    if avg_response_hours is None:
        return 0.95
    if avg_response_hours <= 2:
        return 1.0
    elif avg_response_hours <= 24:
        return 0.95
    elif avg_response_hours <= 72:
        return 0.85
    else:
        return 0.70


def activity_signal_multiplier(
    applications_30d: Optional[int],
    profile_views_30d: Optional[int],
    saved_by_recruiters_30d: Optional[int],
) -> float:
    """Active applicants with profile views and recruiter interest are more engaged."""
    mult = 1.0
    
    # Applications submitted in last 30 days = active job search
    if applications_30d is not None:
        if applications_30d >= 5:
            mult *= 1.05
        elif applications_30d == 0:
            mult *= 0.85
    
    # Profile views = market-validated demand
    if profile_views_30d is not None:
        if profile_views_30d >= 10:
            mult *= 1.05
        elif profile_views_30d == 0:
            mult *= 0.95
    
    # Saved by recruiters = direct interest signal
    if saved_by_recruiters_30d is not None:
        if saved_by_recruiters_30d >= 3:
            mult *= 1.05
        elif saved_by_recruiters_30d == 0:
            mult *= 0.90
    
    return round(mult, 4)


def offer_acceptance_multiplier(acceptance_rate: Optional[float]) -> float:
    """
    -1 = no prior offers (neutral)
    Low (0.0-0.3) = flight risk (down-weight)
    High (0.8+) = committed (up-weight)
    """
    if acceptance_rate is None or acceptance_rate == -1:
        return 1.0
    if acceptance_rate >= 0.80:
        return 1.05
    elif acceptance_rate >= 0.50:
        return 1.0
    elif acceptance_rate >= 0.20:
        return 0.85
    else:
        return 0.70


def network_signal_multiplier(
    connection_count: Optional[int],
    endorsements: Optional[int],
) -> float:
    """Network size and peer validation signals."""
    mult = 1.0
    
    if connection_count is not None:
        if connection_count >= 1000:
            mult *= 1.05
        elif connection_count >= 500:
            mult *= 1.02
        elif connection_count < 50:
            mult *= 0.95
    
    if endorsements is not None:
        if endorsements >= 100:
            mult *= 1.05
        elif endorsements >= 50:
            mult *= 1.02
        elif endorsements == 0:
            mult *= 0.95
    
    return round(mult, 4)


# ─── JD DISQUALIFIER CHECKS ─────────────────────────────────────────────────

def _check_consulting_only_career(record: dict) -> bool:
    """JD disqualifier: Career exclusively at consulting firms."""
    career = record.get("career_history") or []
    if len(career) < 3:
        return False
    
    consulting_count = 0
    for job in career:
        if isinstance(job, dict):
            company = (job.get("company") or "").lower()
            if any(trap in company for trap in CONSULTING_TRAP_COMPANIES):
                consulting_count += 1
    
    # All jobs (or nearly all) at consulting firms = trap
    return consulting_count >= len(career) * 0.8


def _check_title_chaser(record: dict) -> bool:
    """JD disqualifier: 'Title-chasers switching every 1.5 years'"""
    career = record.get("career_history") or []
    if len(career) < 3:
        return False
    
    switch_count = 0
    for job in career:
        if isinstance(job, dict):
            duration = job.get("duration_months")
            try:
                dur_months = float(duration) if duration else 0
                if dur_months < 18:  # Less than 1.5 years
                    switch_count += 1
            except (TypeError, ValueError):
                pass
    
    # Multiple quick switches = title chaser
    return switch_count >= len(career) * 0.6


def _check_pure_research_no_production(record: dict) -> bool:
    """JD disqualifier: 'Pure research without production deployment'"""
    text = (record.get("summary") or "") + " " + (record.get("profile", {}).get("current_title") or "")
    title_lower = (record.get("profile", {}).get("current_title") or "").lower()
    
    research_signals = sum(1 for sig in PURE_RESEARCH_SIGNALS if sig in title_lower)
    
    # Check if there's explicit production/deployment language
    production_signals = {
        "shipped", "deployed", "production", "live users", "scaled",
        "built product", "launched", "released", "serving",
    }
    has_production = any(sig in text.lower() for sig in production_signals)
    
    # If senior researcher/scientist without production claims = flag
    if research_signals >= 2 and not has_production:
        return True
    
    return False


def _check_senior_no_code(record: dict, years: Optional[float]) -> bool:
    """JD disqualifier: 'Senior (5+ YOE) hasn't written code in 18 months'"""
    if not years or years < 5:
        return False
    
    career = record.get("career_history") or []
    if not career:
        return False
    
    # Check most recent role for code-related activity
    recent_job = career[0] if career else {}
    description = (recent_job.get("description") or "").lower()
    end_date = recent_job.get("end_date")
    
    # If recent job (within 18 months) doesn't mention code/development
    code_signals = {
        "code", "develop", "engineer", "build", "implement",
        "architecture", "design", "python", "java", "c++",
    }
    has_code = any(sig in description for sig in code_signals)
    
    if not has_code:
        try:
            if end_date:
                end_year = int(end_date[:4])
                current_year = datetime.now().year
                if current_year - end_year <= 1.5:
                    return True
        except (ValueError, IndexError):
            pass
    
    return False


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
    their current day-to-day work.
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
    NOW INCLUDES: All 16 behavioral signals + JD disqualifiers.
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

    # ─── NEW: BEHAVIORAL SIGNAL EXTRACTION AND MULTIPLIERS ──────────────────
    open_to_work = _get_open_to_work_flag(record)
    notice_days = _get_notice_period_days(record)
    work_mode = _get_preferred_work_mode(record)
    willing_relocate = _get_willing_to_relocate(record)
    expected_salary = _get_expected_salary_inr_lpa(record)
    avg_response_hrs = _get_avg_response_time_hours(record)
    apps_30d = _get_applications_submitted_30d(record)
    views_30d = _get_profile_views_received_30d(record)
    saved_30d = _get_saved_by_recruiters_30d(record)
    acceptance_rate = _get_offer_acceptance_rate(record)
    connections = _get_connection_count(record)
    endorsements = _get_endorsements_received(record)
    location, country = _get_location_info(record)
    education_tier = _get_education_tier(record)

    # ─── BEHAVIORAL SIGNAL MULTIPLIERS ──────────────────────────────────────
    mult_open_to_work = open_to_work_multiplier(open_to_work)
    mult_notice = notice_period_multiplier(notice_days)
    mult_work_mode = work_mode_multiplier(work_mode)
    mult_relocate = relocation_multiplier(willing_relocate, location)
    mult_response_time = response_time_multiplier(avg_response_hrs)
    mult_activity = activity_signal_multiplier(apps_30d, views_30d, saved_30d)
    mult_acceptance = offer_acceptance_multiplier(acceptance_rate)
    mult_network = network_signal_multiplier(connections, endorsements)

    # Combine all behavioral multipliers
    behavioral_mult = (
        mult_open_to_work
        * mult_notice
        * mult_work_mode
        * mult_relocate
        * mult_response_time
        * mult_activity
        * mult_acceptance
        * mult_network
    )

    # ─── JD DISQUALIFIERS ───────────────────────────────────────────────────
    is_consulting_only = _check_consulting_only_career(record)
    is_title_chaser = _check_title_chaser(record)
    is_pure_research = _check_pure_research_no_production(record)
    is_senior_no_code = _check_senior_no_code(record, years)

    disqualifier_flags = []
    disqualifier_mult = 1.0
    if is_consulting_only:
        disqualifier_flags.append("Consulting-only career (JD disqualifier)")
        disqualifier_mult *= 0.30
    if is_title_chaser:
        disqualifier_flags.append("Title-chaser pattern: frequent job switches < 1.5 years")
        disqualifier_mult *= 0.40
    if is_pure_research:
        disqualifier_flags.append("Pure research without production deployment (JD disqualifier)")
        disqualifier_mult *= 0.35
    if is_senior_no_code:
        disqualifier_flags.append("Senior engineer but no code written in 18+ months")
        disqualifier_mult *= 0.50

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

    # Apply honeypot, stuffer, engagement penalties
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

    # Apply behavioral signals and disqualifiers
    final_score *= behavioral_mult
    final_score *= disqualifier_mult

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
        f"Responsive: {avg_response_hrs:.1f}h avg response time" if avg_response_hrs and avg_response_hrs <= 24 else "",
        f"Location: {location} — {('preferred area' if any(city in location.lower() for city in PREFERRED_LOCATIONS) else 'willing to relocate')}" if location else "",
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
        disqualifier_flags[0] if disqualifier_flags else "",
        "Not open to work" if not open_to_work else "",
        f"Long notice period: {notice_days} days" if notice_days and notice_days > 60 else "",
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
    
        "open_to_work": open_to_work,
        "notice_period_days": notice_days,
        "preferred_work_mode": work_mode,
        "willing_to_relocate": willing_relocate,
        "expected_salary_inr_lpa": expected_salary,
        "avg_response_time_hours": avg_response_hrs,
        "applications_30d": apps_30d,
        "profile_views_30d": views_30d,
        "saved_by_recruiters_30d": saved_30d,
        "offer_acceptance_rate": acceptance_rate,
        "connection_count": connections,
        "endorsements": endorsements,
        "location": location,
        "country": country,
        "education_tier": education_tier,
        "behavioral_multiplier": behavioral_mult,
    
        "disqualifiers": disqualifier_flags,
        "disqualifier_multiplier": disqualifier_mult,
        
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