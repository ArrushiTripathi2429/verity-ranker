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


def buzzword_density(text: str) -> float:
    if not text:
        return 0.0
    words = max(len(text.split()), 1)
    lowered = text.lower()
    hits = sum(1 for bw in BUZZWORDS if bw in lowered)
    return hits / words


def honeypot_risk(record: dict[str, Any]) -> tuple[float, list[str]]:
    """Return risk score 0-1 and human-readable flags."""
    text = resume_text(record)
    title = job_title(record)
    skills = skills_list(record)
    years = years_experience(record)
    flags: list[str] = []
    risk = 0.0

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

    return min(1.0, risk), flags
