"""
BM25 lexical retrieval for Layer 8.

Scores every candidate against the JD's skill terms using BM25 — the
gold-standard probabilistic term-frequency model. BM25 excels at exact
keyword and tool name matches that dense embeddings can miss (e.g. a
specific library version, a proprietary tool name, an acronym).

Two execution modes:
  1. rank-bm25 mode  — uses the BM25Okapi implementation from the
                        rank-bm25 library (pip install rank-bm25).
  2. Fallback mode   — pure Python TF-IDF approximation that works
                        without any extra dependencies.

Public API
----------
score_bm25(candidates, jd_terms) → dict[candidate_id, float]
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
     "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
     "i", "my", "me", "we", "our", "you", "your", "it", "its"]
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words."""
    tokens = re.findall(r"[a-z0-9+#/.-]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _profile_to_doc(profile: CandidateProfile) -> str:
    """
    Flatten a CandidateProfile into a single text document for BM25 indexing.

    Skill names are repeated 3× to give them higher term weight — they're
    the most signal-dense part of the profile.
    """
    parts: list[str] = []

    # Skills — high weight, repeated
    for skill in profile.skills:
        parts.extend([skill.skill] * 3)
        parts.extend(skill.evidence_snippets)

    # Projects
    for proj in profile.projects:
        parts.append(proj.title)
        parts.append(proj.description)
        parts.extend(proj.skills_used)

    # Achievements
    for ach in profile.achievements:
        parts.append(ach.description)
        if ach.metric_snippet:
            parts.append(ach.metric_snippet)

    # Career timeline
    for role in profile.career_timeline:
        parts.append(role.title)
        parts.extend(role.responsibilities)

    # Certifications
    parts.extend(profile.certifications)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# rank-bm25 implementation
# ---------------------------------------------------------------------------


def _score_bm25_library(
    corpus: list[list[str]],
    query: list[str],
) -> list[float]:
    """Use rank-bm25 if available."""
    from rank_bm25 import BM25Okapi  # type: ignore
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)
    return list(scores)


# ---------------------------------------------------------------------------
# Pure Python fallback — TF-IDF approximation
# ---------------------------------------------------------------------------


def _score_tfidf_fallback(
    corpus: list[list[str]],
    query: list[str],
) -> list[float]:
    """
    Simple TF-IDF approximation of BM25.

    Formula:
      score(d, q) = Σ_t tf(t,d) × idf(t) × query_weight(t)

    where:
      tf(t,d)  = count(t in d) / len(d)      (normalised term frequency)
      idf(t)   = log((N+1) / (df(t)+1)) + 1  (smoothed IDF)
    """
    N = len(corpus)
    if N == 0:
        return []

    # Document frequencies
    df: Counter[str] = Counter()
    for doc in corpus:
        df.update(set(doc))

    def idf(term: str) -> float:
        return math.log((N + 1) / (df.get(term, 0) + 1)) + 1.0

    query_terms = set(query)
    scores: list[float] = []

    for doc in corpus:
        if not doc:
            scores.append(0.0)
            continue
        doc_len = len(doc)
        term_counts = Counter(doc)
        score = sum(
            (term_counts[t] / doc_len) * idf(t)
            for t in query_terms
            if t in term_counts
        )
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def score_bm25(
    candidates: list[CandidateProfile],
    jd_terms: list[str],
) -> dict[str, float]:
    """
    Score every candidate against JD skill terms using BM25 (or TF-IDF fallback).

    Scores are normalised to [0, 1] by dividing by the maximum score in the
    batch. A candidate that perfectly matches all JD terms gets 1.0.

    Args:
        candidates:  CandidateProfiles from Layer 4.
        jd_terms:    Skill / keyword terms from the JD (required + preferred).
                     These are the BM25 query tokens.

    Returns:
        dict mapping candidate_id → normalised BM25 score (0–1).
    """
    if not candidates or not jd_terms:
        return {c.candidate_id: 0.0 for c in candidates}

    # Build corpus
    docs = [_profile_to_doc(c) for c in candidates]
    corpus = [_tokenise(doc) for doc in docs]
    query  = _tokenise(" ".join(jd_terms))

    if not query:
        return {c.candidate_id: 0.0 for c in candidates}

    # Score
    try:
        raw_scores = _score_bm25_library(corpus, query)
        logger.debug("BM25 scored %d candidates (rank-bm25 library)", len(candidates))
    except ImportError:
        raw_scores = _score_tfidf_fallback(corpus, query)
        logger.debug("BM25 scored %d candidates (TF-IDF fallback)", len(candidates))

    # Normalise to [0, 1]
    max_score = max(raw_scores) if raw_scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    return {
        c.candidate_id: round(raw_scores[i] / max_score, 4)
        for i, c in enumerate(candidates)
    }
