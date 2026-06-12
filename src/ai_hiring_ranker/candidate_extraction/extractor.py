"""
Candidate Profile Extractor — Layer 4.

Converts a CandidateInput (raw resume text) into a fully structured
CandidateProfile consumed by every downstream layer.

Two modes:
  1. LLM mode  — rich, contextual extraction via structured_completion().
  2. Fallback  — deterministic NER-style rule extraction via regex + keyword
                 lists. Covers all fields; less accurate than LLM on complex
                 resumes, but always works offline.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from ..ingestion.schemas import CandidateInput
from ..llm_provider import structured_completion
from .schemas import (
    Achievement,
    CareerRole,
    CandidateProfile,
    DegreeLevel,
    EducationEntry,
    EmploymentCategory,
    ProjectEntry,
    SkillClaim,
    SkillConfidence,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[4] / "prompts" / "candidate_extraction.md"




def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a resume analyst. Extract a structured JSON candidate profile "
        "from the resume text. Return ONLY valid JSON."
    )




_SKILL_ALIASES: dict[str, list[str]] = {
    "Python":               ["python"],
    "Machine Learning":     ["machine learning", " ml ", "ml,", "ml."],
    "Deep Learning":        ["deep learning", "neural network", "pytorch", "tensorflow", "keras"],
    "Fastapi":              ["fastapi", "fast api"],
    "Docker":               ["docker", "containerization", "containers"],
    "Kubernetes":           ["kubernetes", "k8s"],
    "Sql":                  ["sql", "postgres", "mysql", "sqlite", "postgresql"],
    "Nosql":                ["nosql", "mongodb", "redis", "dynamodb", "cassandra"],
    "Rest Api":             ["rest api", "restful", "api design", "http api"],
    "Cloud":                ["aws", "gcp", "azure", "cloud deployment", "cloud infrastructure"],
    "Ci/Cd":                ["ci/cd", "ci pipeline", "github actions", "jenkins", "gitlab ci"],
    "Model Evaluation":     ["model evaluation", "evaluation pipeline", "metrics", "f1", "auc", "accuracy"],
    "Embeddings":           ["embedding", "vector", "sentence transformer"],
    "Retrieval":            ["retrieval", "vector search", "faiss", "bm25", "semantic search"],
    "Llm":                  ["llm", "large language model", "gpt", "language model", "chatgpt"],
    "Langchain":            ["langchain", "langgraph", "langsmith"],
    "Production Engineering":["production", "production-ready", "production system", "reliability"],
    "Testing":              ["unit test", "pytest", "testing", "test coverage", "tdd"],
    "Git":                  ["git ", "github", "version control", "gitlab"],
    "Spark":                ["spark", "pyspark", "distributed computing"],
    "Airflow":              ["airflow", "workflow orchestration", "dag"],
    "Mlflow":               ["mlflow", "experiment tracking", "model registry"],
    "Nlp":                  ["nlp", "natural language processing", "text classification", "named entity", "spacy", "nltk"],
    "Computer Vision":      ["computer vision", "image classification", "object detection", "opencv"],
    "Statistics":           ["statistics", "statistical", "probability", "hypothesis testing", "bayesian"],
    "Data Engineering":     ["data pipeline", "etl", "data engineering", "data warehouse"],
    "Scikit-Learn":         ["scikit-learn", "sklearn"],
    "Pandas":               ["pandas", "dataframe"],
    "Numpy":                ["numpy", "np."],
    "Visualization":        ["matplotlib", "seaborn", "plotly", "tableau", "power bi", "dashboard"],
}

# Negation patterns — skip sentences that deny a skill
_NEGATION_RE = re.compile(
    r"\b(no |not |without |lack |limited |never |haven't |don't have )\b",
    re.I,
)

# Seniority keyword → signal value
_SENIORITY_SIGNALS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bprincipal\b|\bstaff\b",      re.I), 1.0),
    (re.compile(r"\blead\b|\barchitect\b",        re.I), 0.9),
    (re.compile(r"\bsenior\b|\bsr\.?\b",          re.I), 0.8),
    (re.compile(r"\b[6-9]\s*\+?\s*years?\b|\b1[0-9]\s*\+?\s*years?\b", re.I), 0.85),
    (re.compile(r"\b[3-5]\s*\+?\s*years?\b",      re.I), 0.55),
    (re.compile(r"\bjunior\b|\bjr\.?\b",          re.I), 0.25),
    (re.compile(r"\bintern\b",                    re.I), 0.10),
]

_LEADERSHIP_KEYWORDS  = ["led", "lead", "managed", "owned", "drove", "architected",
                          "mentored", "coordinated", "directed", "founded", "established"]
_PRODUCTION_KEYWORDS  = ["production", "deployed", "live", "served", "released",
                          "launched", "shipped", "ran in production", "cloud deployment"]
_ACHIEVEMENT_KEYWORDS = ["reduced", "improved", "increased", "achieved", "cut",
                          "built", "created", "delivered", "saved", "accelerated"]
_METRIC_RE            = re.compile(r"\d+\s*(%|x\b|ms\b|users?\b|requests?\b|latency|throughput|accuracy)", re.I)
_YEAR_RE              = re.compile(r"\b(20[0-2]\d|19[89]\d)\b")
_EMAIL_RE             = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE             = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")

# Role title markers for career timeline detection
_ROLE_TITLE_RE = re.compile(
    r"(?:^|\n)\s*([A-Z][A-Za-z\s/]+(?:Engineer|Scientist|Developer|Analyst|Architect|"
    r"Manager|Lead|Director|Intern|Researcher|Specialist|Consultant|Head))"
    r"(?:\s+at\s+|\s*[-–|]\s*|\s*,\s*)([A-Za-z0-9 &.,]+)",
    re.MULTILINE,
)

# Education markers
_DEGREE_MAP: list[tuple[re.Pattern, DegreeLevel]] = [
    (re.compile(r"\bph\.?d\b|\bdoctor",                re.I), DegreeLevel.PHD),
    (re.compile(r"\bm\.?s\.?\b|\bmaster",              re.I), DegreeLevel.MASTERS),
    (re.compile(r"\bb\.?s\.?\b|\bbachelor|\bb\.?tech\b|\bb\.?e\b", re.I), DegreeLevel.BACHELORS),
    (re.compile(r"\bassociate",                        re.I), DegreeLevel.ASSOCIATE),
    (re.compile(r"\bbootcamp\b",                       re.I), DegreeLevel.BOOTCAMP),
    (re.compile(r"\bself.?taught\b|\baudodidact",      re.I), DegreeLevel.SELF_TAUGHT),
]

_CERT_KEYWORDS = [
    "certified", "certification", "certificate", "aws certified",
    "google certified", "azure certified", "coursera", "udacity",
    "deeplearning.ai", "cka", "ckad", "pmp",
]

# Sentence splitter
def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]





def _extract_name(text: str) -> str:
    match = re.search(r"^\s*Name\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try first non-empty line that looks like a name (Title Case, ≤40 chars, no colons)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= 40 and ":" not in stripped and stripped.istitle():
            return stripped
    return ""


def _extract_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> Optional[str]:
    m = _PHONE_RE.search(text)
    raw = m.group(0).strip() if m else None
    # Ignore short numbers that are likely years
    if raw and len(re.sub(r"\D", "", raw)) >= 7:
        return raw
    return None


def _extract_skills(text: str, sentences: list[str]) -> list[SkillClaim]:
    """Rule-based skill extraction with evidence snippets."""
    claims: list[SkillClaim] = []
    seen: set[str] = set()
    lowered = text.lower()

    # First pass: explicit Skills: line
    skills_line_match = re.search(r"^Skills\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    explicit_skills: set[str] = set()
    if skills_line_match:
        parts = re.split(r"[,;]", skills_line_match.group(1))
        for part in parts:
            part = part.strip().title()
            if part:
                explicit_skills.add(part)

    for skill_name, aliases in _SKILL_ALIASES.items():
        if skill_name in seen:
            continue
        if not any(alias in lowered for alias in aliases):
            continue
        seen.add(skill_name)

        # Collect evidence from non-negated sentences
        evidence: list[str] = []
        for sentence in sentences:
            if _NEGATION_RE.search(sentence):
                continue
            sent_lower = sentence.lower()
            if any(alias in sent_lower for alias in aliases):
                evidence.append(sentence[:200])
            if len(evidence) == 3:
                break

        # Classify confidence
        if skill_name in explicit_skills or skill_name.lower() in {e.lower() for e in explicit_skills}:
            confidence = SkillConfidence.EXPLICIT
        elif len(evidence) >= 2:
            confidence = SkillConfidence.EXPLICIT
        elif len(evidence) == 1:
            confidence = SkillConfidence.INFERRED
        else:
            confidence = SkillConfidence.WEAK

        # Try to extract years of experience
        years: Optional[float] = None
        for snippet in evidence:
            m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", snippet, re.I)
            if m:
                years = float(m.group(1))
                break

        # Try to find last used year
        last_year: Optional[int] = None
        for snippet in evidence:
            matches = _YEAR_RE.findall(snippet)
            if matches:
                last_year = max(int(y) for y in matches)
                break

        claims.append(SkillClaim(
            skill=skill_name,
            confidence=confidence,
            evidence_snippets=evidence,
            years_of_experience=years,
            last_used_year=last_year,
        ))

    return claims


def _extract_projects(sentences: list[str]) -> list[ProjectEntry]:
    """Heuristic: sentences with build/create/develop + a noun phrase."""
    project_verbs = re.compile(
        r"\b(built|created|developed|designed|implemented|wrote|built|deployed|released)\b",
        re.I,
    )
    projects: list[ProjectEntry] = []
    for sentence in sentences:
        if not project_verbs.search(sentence):
            continue
        if len(sentence) < 20:
            continue
        skills_used = [
            name for name, aliases in _SKILL_ALIASES.items()
            if any(a in sentence.lower() for a in aliases)
        ]
        is_prod = any(kw in sentence.lower() for kw in _PRODUCTION_KEYWORDS)
        has_metrics = bool(_METRIC_RE.search(sentence))
        projects.append(ProjectEntry(
            title=sentence[:60].rstrip("."),
            description=sentence,
            skills_used=skills_used,
            is_production=is_prod,
            has_metrics=has_metrics,
            source_snippet=sentence[:200],
        ))
        if len(projects) == 6:  # cap
            break
    return projects


def _extract_achievements(sentences: list[str]) -> list[Achievement]:
    achievements: list[Achievement] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(kw in lowered for kw in _ACHIEVEMENT_KEYWORDS):
            continue
        has_metric = bool(_METRIC_RE.search(sentence))
        metric_snippet: Optional[str] = None
        if has_metric:
            m = _METRIC_RE.search(sentence)
            if m:
                start = max(0, m.start() - 20)
                metric_snippet = sentence[start : m.end() + 10].strip()
        achievements.append(Achievement(
            description=sentence[:300],
            has_metric=has_metric,
            metric_snippet=metric_snippet,
            source_snippet=sentence[:200],
        ))
        if len(achievements) == 8:
            break
    return achievements


def _extract_career_timeline(text: str) -> list[CareerRole]:
    roles: list[CareerRole] = []
    for match in _ROLE_TITLE_RE.finditer(text):
        title   = match.group(1).strip()
        company = match.group(2).strip()[:60]
        years   = [int(y) for y in _YEAR_RE.findall(match.group(0))]
        start   = min(years) if years else None
        end     = max(years) if len(years) >= 2 else None
        roles.append(CareerRole(
            title=title,
            company=company,
            start_year=start,
            end_year=end,
            category=EmploymentCategory.FULL_TIME,
        ))
    return roles


def _extract_education(text: str) -> list[EducationEntry]:
    entries: list[EducationEntry] = []
    for line in text.splitlines():
        lowered = line.lower()
        level = DegreeLevel.UNKNOWN
        for pattern, deg_level in _DEGREE_MAP:
            if pattern.search(lowered):
                level = deg_level
                break
        if level == DegreeLevel.UNKNOWN:
            continue
        years = _YEAR_RE.findall(line)
        year  = int(years[-1]) if years else None
        entries.append(EducationEntry(
            degree=line.strip()[:100],
            level=level,
            year=year,
        ))
    return entries[:3]  # cap at 3


def _extract_certifications(sentences: list[str]) -> list[str]:
    certs: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(kw in lowered for kw in _CERT_KEYWORDS):
            certs.append(sentence.strip()[:150])
    return certs[:5]


def _compute_seniority_signal(text: str) -> float:
    for pattern, value in _SENIORITY_SIGNALS:
        if pattern.search(text):
            return value
    return 0.3  # default mid


def _compute_leadership_signal(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for kw in _LEADERSHIP_KEYWORDS if kw in lowered)
    return min(hits / 3.0, 1.0)


def _compute_production_signal(sentences: list[str]) -> float:
    hits = sum(
        1 for s in sentences
        if any(kw in s.lower() for kw in _PRODUCTION_KEYWORDS)
        and not _NEGATION_RE.search(s)
    )
    return min(hits / 3.0, 1.0)


def _compute_achievement_signal(achievements: list[Achievement]) -> float:
    if not achievements:
        return 0.0
    with_metrics = sum(1 for a in achievements if a.has_metric)
    base = min(len(achievements) / 4.0, 0.5)
    metric_bonus = min(with_metrics / 3.0, 0.5)
    return round(base + metric_bonus, 2)


def _compute_career_growth_signal(timeline: list[CareerRole]) -> float:
    """Simple heuristic: more roles + title progression → higher signal."""
    if len(timeline) < 2:
        return 0.3
    # Check for upward title keywords
    progression_keywords = ["senior", "lead", "principal", "staff", "head", "director", "manager"]
    has_progression = any(
        any(kw in role.title.lower() for kw in progression_keywords)
        for role in timeline
    )
    base = min(len(timeline) / 4.0, 0.5)
    bonus = 0.5 if has_progression else 0.0
    return round(min(base + bonus, 1.0), 2)


def _estimate_total_years(timeline: list[CareerRole], text: str) -> Optional[float]:
    # Sum durations from timeline
    total = sum(r.duration_years for r in timeline if r.duration_years is not None)
    if total > 0:
        return round(total, 1)
    # Fall back to highest year mention in the resume
    m = re.search(r"(\d+)\s*\+?\s*years?\s+of\s+experience", text, re.I)
    if m:
        return float(m.group(1))
    return None




def _run_fallback(candidate: CandidateInput) -> CandidateProfile:
    text      = candidate.raw_text
    sents     = _sentences(text)

    skills    = _extract_skills(text, sents)
    projects  = _extract_projects(sents)
    achievements = _extract_achievements(sents)
    timeline  = _extract_career_timeline(text)
    education = _extract_education(text)
    certs     = _extract_certifications(sents)

    seniority_sig  = _compute_seniority_signal(text)
    leadership_sig = _compute_leadership_signal(text)
    production_sig = _compute_production_signal(sents)
    achievement_sig = _compute_achievement_signal(achievements)
    growth_sig     = _compute_career_growth_signal(timeline)
    total_years    = _estimate_total_years(timeline, text)

    return CandidateProfile(
        candidate_id    = candidate.candidate_id,
        name            = _extract_name(text),
        email           = _extract_email(text),
        phone           = _extract_phone(text),
        skills          = skills,
        career_timeline = timeline,
        projects        = projects,
        achievements    = achievements,
        education       = education,
        certifications  = certs,
        total_years_experience = total_years,
        seniority_signal  = seniority_sig,
        leadership_signal = leadership_sig,
        production_signal = production_sig,
        achievement_signal = achievement_sig,
        career_growth_signal = growth_sig,
    )





def _run_llm(candidate: CandidateInput) -> CandidateProfile:
    system_prompt = _load_system_prompt()
    user_prompt   = (
        f"Extract a structured candidate profile from the following resume.\n\n"
        f"Candidate ID: {candidate.candidate_id}\n\n"
        f"---\n{candidate.raw_text}\n---\n\n"
        "Return only the JSON object."
    )
    return structured_completion(
        system_prompt = system_prompt,
        user_prompt   = user_prompt,
        schema        = CandidateProfile,
    )





def extract_candidate_profile(
    candidate: CandidateInput,
    *,
    force_fallback: bool = False,
) -> CandidateProfile:
    """
    Extract a structured CandidateProfile from a CandidateInput.

    Automatically selects LLM mode if an API key is present, otherwise
    falls back to rule-based extraction.

    Args:
        candidate:      Validated CandidateInput from Layer 1.
        force_fallback: Always use rule-based mode (offline / testing).

    Returns:
        A validated CandidateProfile.
    """
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))

    if force_fallback or not has_key:
        mode = "rule-based fallback (no API key)" if not has_key else "rule-based fallback (forced)"
        logger.info("Candidate extractor [%s] running in %s mode", candidate.candidate_id, mode)
        return _run_fallback(candidate)

    logger.info("Candidate extractor [%s] running in LLM mode", candidate.candidate_id)
    try:
        return _run_llm(candidate)
    except Exception as exc:
        logger.warning(
            "LLM extraction failed for %s (%s) — falling back to rule-based.",
            candidate.candidate_id, exc,
        )
        return _run_fallback(candidate)


def extract_all_candidates(
    candidates: list[CandidateInput],
    *,
    force_fallback: bool = False,
) -> list[CandidateProfile]:
    """
    Extract profiles for a full list of candidates.

    Per-candidate failures are logged and skipped rather than aborting the batch.

    Returns:
        List of successfully extracted CandidateProfiles (may be shorter than input).
    """
    profiles: list[CandidateProfile] = []
    for candidate in candidates:
        try:
            profile = extract_candidate_profile(candidate, force_fallback=force_fallback)
            profiles.append(profile)
        except Exception as exc:
            logger.error(
                "Failed to extract profile for %s: %s — skipping.",
                candidate.candidate_id, exc,
            )
    return profiles
