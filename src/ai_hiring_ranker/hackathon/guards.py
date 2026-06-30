"""Honeypot and keyword-stuffer heuristics for hackathon dataset."""

from __future__ import annotations

import re
from typing import Any

from .dataset import job_title, resume_text, skills_list, years_experience

BUZZWORDS = (
    "llm", "large language model", "rag", "retrieval augmented",
    "transformer", "genai", "generative ai", "blockchain", "web3",
    "metaverse", "nft", "quantum", "kubernetes", "microservices",
    "distributed systems", "machine learning", "deep learning",
    "neural network", "nlp", "computer vision", "prompt engineering",
    "langchain", "vector database", "embeddings", "fine-tuning",
)

IMPOSSIBLE_PHRASES = (
    "invented python",
    "created python",
    "nobel prize",
    "50+ years",
    "100 years experience",
    "ceo of google",
    "ceo of microsoft",
    "worked at every faang",
)

TITLE_MISMATCH_PAIRS = (
    (re.compile(r"\bintern\b", re.I), 8.0),
    (re.compile(r"\bstudent\b", re.I), 3.0),
    (re.compile(r"\bjunior\b", re.I), 12.0),
)

CONSULTING_TRAP_COMPANIES = (
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "deloitte", "pwc", "kpmg", "ibm consulting", "mckinsey",
)


def buzzword_density(text: str) -> float:
    if not text:
        return 0.0
    words = max(len(text.split()), 1)
    lowered = text.lower()
    hits = sum(1 for bw in BUZZWORDS if bw in lowered)
    return hits / words


def _extract_career_history(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Safely extract career_history array from record."""
    career = record.get("career_history", [])
    if isinstance(career, list):
        return career
    return []


def _extract_skills(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Safely extract skills array from record."""
    skills = record.get("skills", [])
    if isinstance(skills, list):
        return skills
    return []


def _check_skill_duration_contradiction(skills: list[dict[str, Any]]) -> tuple[bool, str]:
    """
    CRITICAL HONEYPOT CHECK: "expert proficiency in 10 skills with 0 duration_months"
    
    Returns: (is_honeypot, reason)
    """
    expert_zero_duration = []
    
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        
        proficiency = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months")
        
        # Expert level with 0 or missing duration = red flag
        if proficiency == "expert":
            try:
                dur_val = float(duration) if duration is not None else 0
                if dur_val == 0 or duration is None:
                    expert_zero_duration.append(skill.get("name", "unknown"))
            except (TypeError, ValueError):
                # Can't parse duration, treat as 0
                if duration is None or str(duration).strip() == "":
                    expert_zero_duration.append(skill.get("name", "unknown"))
    
    if len(expert_zero_duration) >= 3:  # Multiple expert skills with no duration = honeypot
        return True, f"Expert proficiency in {len(expert_zero_duration)} skills with 0 duration: {', '.join(expert_zero_duration[:5])}"
    
    return False, ""


def _check_title_description_mismatch(career_history: list[dict[str, Any]]) -> tuple[bool, str]:
    """
    HONEYPOT CHECK: Role description contradicts job title.
    E.g., "Marketing Manager" whose description talks about "Mechanical engineering design"
    
    Returns: (is_honeypot, reason)
    """
    mismatches = []
    
    for job in career_history:
        if not isinstance(job, dict):
            continue
        
        title = (job.get("title") or "").lower()
        description = (job.get("description") or "").lower()
        
        if not title or not description:
            continue
        
        # Extract domain keywords from title
        title_keywords = set()
        if any(kw in title for kw in ("marketing", "sales", "business", "management")):
            title_keywords.add("business")
        if any(kw in title for kw in ("engineer", "developer", "architect", "technical")):
            title_keywords.add("engineering")
        if any(kw in title for kw in ("design", "ui", "ux", "visual")):
            title_keywords.add("design")
        if any(kw in title for kw in ("data", "analytics", "analyst")):
            title_keywords.add("data")
        
        # Extract domain keywords from description
        description_keywords = set()
        if any(kw in description for kw in ("mechanical", "civil", "structural", "thermodynamics")):
            description_keywords.add("mechanical")
        if any(kw in description for kw in ("marketing", "campaign", "brand", "outreach")):
            description_keywords.add("marketing")
        if any(kw in description for kw in ("engineering", "design", "build", "develop")):
            description_keywords.add("engineering")
        
        # Check for contradictions: if title says "marketing" but description says "mechanical engineering"
        if title_keywords and description_keywords:
            if title_keywords.isdisjoint(description_keywords):
                mismatches.append(f"Title '{job.get('title')}' ≠ Description domain")
    
    if len(mismatches) >= 2:
        return True, f"Multiple title-description mismatches: {'; '.join(mismatches[:3])}"
    
    return False, ""


def _check_company_tenure_impossibility(career_history: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """
    HONEYPOT CHECK: "8 years at company founded 3 years ago"
    
    Requires cross-checking start_date and duration against company founding dates.
    We implement a simplified version: if duration_months > (now - start_date),
    it's impossible.
    
    Returns: (is_impossible, list_of_issues)
    """
    from datetime import datetime
    
    issues = []
    now = datetime.now()
    
    for job in career_history:
        if not isinstance(job, dict):
            continue
        
        duration = job.get("duration_months")
        start_date = job.get("start_date")  # Assume ISO format or similar
        
        if duration is None or start_date is None:
            continue
        
        try:
            duration_months = float(duration)
            
            # Try to parse start_date (simple heuristic: if it's a string like "2020-01", extract year)
            if isinstance(start_date, str):
                # Extract year from start_date (e.g., "2020-01" or "2020")
                year_match = re.search(r"(20\d{2})", start_date)
                if year_match:
                    start_year = int(year_match.group(1))
                    current_year = now.year
                    max_possible_months = (current_year - start_year) * 12
                    
                    if duration_months > max_possible_months + 12:  # Allow 1 year buffer
                        issues.append(
                            f"Job at '{job.get('company', 'unknown')}' claims {duration_months} months "
                            f"starting {start_year}, but only {max_possible_months} months could have elapsed"
                        )
        except (TypeError, ValueError):
            continue
    
    return len(issues) >= 1, issues


def _check_summary_vs_career_mismatch(record: dict[str, Any], career_history: list[dict[str, Any]]) -> tuple[bool, str]:
    """
    HONEYPOT CHECK: Summary says "marketing manager roles" but actual titles are "Accountant"
    
    Returns: (is_honeypot, reason)
    """
    summary = (record.get("summary") or "").lower()
    
    if not summary or len(summary) < 20:
        return False, ""
    
    # Extract role keywords from summary
    summary_roles = set()
    if any(kw in summary for kw in ("manager", "lead", "head")):
        summary_roles.add("management")
    if any(kw in summary for kw in ("engineer", "developer", "architect")):
        summary_roles.add("engineering")
    if any(kw in summary for kw in ("marketing", "sales", "business")):
        summary_roles.add("business")
    if any(kw in summary for kw in ("data", "analytics", "scientist")):
        summary_roles.add("data")
    if any(kw in summary for kw in ("accountant", "finance", "accounting")):
        summary_roles.add("finance")
    
    # Extract actual job titles
    actual_titles = []
    for job in career_history:
        if isinstance(job, dict):
            title = (job.get("title") or "").lower()
            if title:
                actual_titles.append(title)
    
    # Check for contradictions
    if summary_roles and actual_titles:
        # Extract role types from actual titles
        title_roles = set()
        for title in actual_titles:
            if any(kw in title for kw in ("accountant", "finance", "accounting", "cfo", "cpa")):
                title_roles.add("finance")
            if any(kw in title for kw in ("engineer", "developer", "architect")):
                title_roles.add("engineering")
            if any(kw in title for kw in ("manager", "lead", "head", "director")):
                title_roles.add("management")
        
        # If summary claims one domain but titles show different domain
        if summary_roles and title_roles and summary_roles.isdisjoint(title_roles):
            return True, f"Summary claims {summary_roles} but actual titles are {title_roles}"
    
    return False, ""


def honeypot_risk(record: dict[str, Any]) -> tuple[float, list[str]]:
    """Return risk score 0-1 and human-readable flags.
    
    Combines heuristic checks with behavioral signal contradictions.
    NOW INCLUDES: skill duration contradictions, title-description mismatches,
    company tenure impossibilities, summary-career mismatches.
    """
    from .dataset import (
        profile_completeness,
        skill_assessment_scores,
        interview_completion_rate,
        verified_contact,
    )
    
    text = resume_text(record)
    title = job_title(record)
    skills = skills_list(record)
    years = years_experience(record)
    flags: list[str] = []
    risk = 0.0
    
    # ─── NEW HONEYPOT CHECKS (from gap analysis) ───────────────
    
    # CHECK 1: Skill proficiency vs duration contradiction
    career = _extract_career_history(record)
    skills_array = _extract_skills(record)
    
    is_skill_trap, reason = _check_skill_duration_contradiction(skills_array)
    if is_skill_trap:
        risk += 0.40
        flags.append(f"🔴 {reason}")
    
    # CHECK 2: Title vs Description mismatch in career history
    is_title_mismatch, reason = _check_title_description_mismatch(career)
    if is_title_mismatch:
        risk += 0.35
        flags.append(f"🔴 {reason}")
    
    # CHECK 3: Company tenure impossibility
    is_tenure_impossible, tenure_issues = _check_company_tenure_impossibility(career)
    if is_tenure_impossible:
        risk += 0.35
        for issue in tenure_issues:
            flags.append(f"🔴 {issue}")
    
    # CHECK 4: Summary vs career title contradiction
    is_summary_mismatch, reason = _check_summary_vs_career_mismatch(record, career)
    if is_summary_mismatch:
        risk += 0.30
        flags.append(f"🔴 {reason}")
    
    # ─── ORIGINAL HEURISTIC CHECKS ───────────────────────────────
    lowered = text.lower()
    for phrase in IMPOSSIBLE_PHRASES:
        if phrase in lowered:
            risk += 0.35
            flags.append(f"impossible claim: '{phrase}'")

    if years is not None and years > 35:
        risk += 0.30
        flags.append(f"implausible experience: {years:.0f} years")

    if len(skills) > 35:
        risk += 0.25
        flags.append(f"skill stuffing: {len(skills)} skills listed")

    if len(text) < 120 and len(skills) > 15:
        risk += 0.20
        flags.append("many skills with very short profile")

    for pattern, max_years in TITLE_MISMATCH_PAIRS:
        if pattern.search(title) and years is not None and years > max_years:
            risk += 0.25
            flags.append(f"title '{title}' conflicts with {years:.0f} years experience")

    gh = record.get("github_activity_score", record.get("github_score"))
    try:
        gh_val = float(gh) if gh is not None else None
        if gh_val is not None and gh_val <= 0.05 and any(
            kw in lowered for kw in ("open source", "github", "maintainer", "contributor")
        ):
            risk += 0.15
            flags.append("claims GitHub activity but github_activity_score is near zero")
    except (TypeError, ValueError):
        pass

    # ─── BEHAVIORAL SIGNAL CONTRADICTIONS ───────────────────────
    completeness = profile_completeness(record)
    if completeness is not None and len(skills) > 10 and completeness < 0.3:
        risk += 0.25
        flags.append(f"Skill overload ({len(skills)} skills) with incomplete profile ({completeness*100:.0f}%)")

    assessments = skill_assessment_scores(record)
    if assessments and len(skills) > 5:
        avg_score = sum(assessments.values()) / len(assessments) if assessments else 0.0
        if avg_score < 0.2:  # < 20% on assessments despite skill claims
            risk += 0.30
            flags.append(f"Skill claims vs assessment mismatch (avg score {avg_score*100:.0f}%)")

    interview_rate = interview_completion_rate(record)
    if interview_rate is not None and interview_rate < 0.2:
        risk += 0.20
        flags.append(f"Low interview completion rate ({interview_rate*100:.0f}%)")

    verified = verified_contact(record)
    if not verified:
        risk += 0.15
        flags.append("No contact verification (email/phone unverified)")

    return min(1.0, risk), flags


def keyword_stuffer_risk(record: dict[str, Any], jd_title: str = "") -> tuple[float, list[str]]:
    text = resume_text(record)
    title = job_title(record)
    skills = skills_list(record)
    flags: list[str] = []
    risk = 0.0

    density = buzzword_density(text)
    if density > 0.12:
        risk += min(0.35, density * 2.0)
        flags.append(f"high AI buzzword density ({density:.2f})")

    if len(skills) >= 20 and len(text.split()) < 180:
        risk += 0.20
        flags.append("keyword-heavy skills list relative to profile length")

    if jd_title:
        jd_tokens = {t for t in re.findall(r"[a-z]{3,}", jd_title.lower())}
        title_tokens = {t for t in re.findall(r"[a-z]{3,}", title.lower())}
        overlap = jd_tokens & title_tokens
        if title and jd_tokens and len(overlap) == 0:
            risk += 0.10
            flags.append(f"job title '{title}' does not align with JD title '{jd_title}'")

    # NEW: Consulting-only career trap
    career = _extract_career_history(record)
    consulting_count = 0
    for job in career:
        if isinstance(job, dict):
            company = (job.get("company") or "").lower()
            if any(trap in company for trap in CONSULTING_TRAP_COMPANIES):
                consulting_count += 1
    
    if consulting_count >= len(career) >= 3:  # All jobs are at consulting firms
        risk += 0.25
        flags.append(f"Career history exclusively at consulting firms ({consulting_count}/{len(career)})")

    return min(1.0, risk), flags