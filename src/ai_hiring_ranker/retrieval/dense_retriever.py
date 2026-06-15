"""
Dense embedding retrieval for Layer 8.

Computes semantic similarity between candidate profiles and:
  a) The JD (via HiringProfile skill terms + responsibilities)
  b) The three HyDE ideal candidate profiles (Layer 3)

Why embed HyDE profiles instead of the JD?
  JDs and resumes live in different embedding spaces — a JD says
  "we need Python expertise" while a resume says "3 years building
  production services in Python". HyDE bridges this gap by generating
  resume-style text for the ideal candidate, so embeddings are
  compared in the same vocabulary space.

Two execution modes:
  1. OpenAI Embeddings — text-embedding-3-small (configured in models.yaml).
     Requires OPENAI_API_KEY. Produces 1536-dim vectors.
  2. Fallback — TF-IDF cosine similarity.
     No network, no API key, no extra dependencies. Lower quality but
     always works and is fast enough for < 1000 candidates.

Public API
----------
score_dense(candidates, jd_profile, hyde_result)
    → dict[candidate_id, float]
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile
from ..config import get_embedding_config
from ..hyde.schemas import HyDEResult
from ..jd_intelligence.schemas import HiringProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    ["a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
     "of", "with", "by", "from", "as", "is", "was", "are", "were"]
)


def _tokenise(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9+#/.-]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _profile_to_text(profile: CandidateProfile) -> str:
    """Convert a CandidateProfile to a flat text for embedding."""
    parts: list[str] = []

    if profile.name:
        parts.append(profile.name)

    for skill in profile.skills:
        parts.append(skill.skill)
        parts.extend(skill.evidence_snippets[:2])

    for proj in profile.projects:
        parts.append(proj.title)
        parts.append(proj.description[:200])

    for ach in profile.achievements:
        parts.append(ach.description[:150])

    for role in profile.career_timeline:
        parts.append(f"{role.title} at {role.company}")
        parts.extend(role.responsibilities[:3])

    parts.extend(profile.certifications)

    return " ".join(p for p in parts if p)


def _jd_to_query_text(hiring_profile: HiringProfile) -> str:
    """Convert HiringProfile to a dense query text."""
    parts: list[str] = [hiring_profile.job_title]
    parts.extend(hiring_profile.all_skill_names)
    parts.extend(hiring_profile.key_responsibilities[:5])
    for he in hiring_profile.hidden_expectations[:3]:
        parts.append(he.description)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# OpenAI embedding
# ---------------------------------------------------------------------------


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the OpenAI Embeddings API."""
    from openai import OpenAI  # lazy import

    cfg = get_embedding_config()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY")
    client = OpenAI(api_key=api_key)

    # Batch in chunks of 100 to avoid hitting the API limit
    all_vectors: list[list[float]] = []
    chunk_size = 100
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        response = client.embeddings.create(
            model=cfg.model,
            input=chunk,
        )
        all_vectors.extend([item.embedding for item in response.data])

    return all_vectors


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _score_dense_openai(
    candidates: list[CandidateProfile],
    jd_query: str,
    hyde_texts: list[str],
) -> dict[str, float]:
    """
    Score candidates using OpenAI embeddings.

    Computes similarity against:
      - The JD query text (weight 0.4)
      - Each of the 3 HyDE profiles (combined weight 0.6, averaged)
    """
    candidate_texts = [_profile_to_text(c) for c in candidates]
    query_texts     = [jd_query] + hyde_texts
    all_texts       = candidate_texts + query_texts

    logger.info("Embedding %d texts via OpenAI API...", len(all_texts))
    all_vectors = _embed_openai(all_texts)

    cand_vecs  = all_vectors[: len(candidates)]
    query_vecs = all_vectors[len(candidates):]

    jd_vec    = query_vecs[0]
    hyde_vecs = query_vecs[1:]

    scores: dict[str, float] = {}
    for i, candidate in enumerate(candidates):
        cv = cand_vecs[i]

        # Similarity to JD
        jd_sim = _cosine(cv, jd_vec)

        # Average similarity to all HyDE profiles
        hyde_sim = sum(_cosine(cv, hv) for hv in hyde_vecs) / len(hyde_vecs) if hyde_vecs else 0.0

        # Blend: HyDE-heavy because it's in the right embedding space
        combined = round(0.40 * jd_sim + 0.60 * hyde_sim, 4)
        scores[candidate.candidate_id] = max(0.0, min(1.0, combined))

    logger.debug("Dense scoring complete (OpenAI): %d candidates", len(candidates))
    return scores


# ---------------------------------------------------------------------------
# TF-IDF cosine fallback (no API key needed)
# ---------------------------------------------------------------------------


def _build_tfidf_vectors(
    docs: list[list[str]],
    vocab: list[str],
) -> list[list[float]]:
    """Build TF-IDF vectors for a corpus given a shared vocabulary."""
    N = len(docs)
    if N == 0:
        return []

    # IDF
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    idf = {
        term: math.log((N + 1) / (df.get(term, 0) + 1)) + 1.0
        for term in vocab
    }

    vectors: list[list[float]] = []
    for doc in docs:
        doc_len = len(doc) or 1
        counts = Counter(doc)
        vec = [(counts.get(t, 0) / doc_len) * idf[t] for t in vocab]
        # L2 normalise
        mag = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / mag for v in vec])

    return vectors


def _score_dense_fallback(
    candidates: list[CandidateProfile],
    jd_query: str,
    hyde_texts: list[str],
) -> dict[str, float]:
    """TF-IDF cosine similarity as a fallback when OpenAI API is unavailable."""
    candidate_texts = [_profile_to_text(c) for c in candidates]
    query_texts     = [jd_query] + hyde_texts
    all_texts       = candidate_texts + query_texts

    tokenised = [_tokenise(t) for t in all_texts]

    # Build shared vocabulary
    vocab_set: set[str] = set()
    for tokens in tokenised:
        vocab_set.update(tokens)
    vocab = sorted(vocab_set)

    all_vectors = _build_tfidf_vectors(tokenised, vocab)
    cand_vecs   = all_vectors[: len(candidates)]
    query_vecs  = all_vectors[len(candidates):]

    jd_vec    = query_vecs[0]
    hyde_vecs = query_vecs[1:]

    scores: dict[str, float] = {}
    for i, candidate in enumerate(candidates):
        cv = cand_vecs[i]
        jd_sim   = _cosine(cv, jd_vec)
        hyde_sim = (
            sum(_cosine(cv, hv) for hv in hyde_vecs) / len(hyde_vecs)
            if hyde_vecs else 0.0
        )
        combined = round(0.40 * jd_sim + 0.60 * hyde_sim, 4)
        scores[candidate.candidate_id] = max(0.0, min(1.0, combined))

    logger.debug("Dense scoring complete (TF-IDF fallback): %d candidates", len(candidates))
    return scores


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def score_dense(
    candidates: list[CandidateProfile],
    hiring_profile: HiringProfile,
    hyde_result: HyDEResult,
    *,
    force_fallback: bool = False,
) -> dict[str, float]:
    """
    Score every candidate by semantic similarity to the JD and HyDE profiles.

    Automatically uses OpenAI embeddings if an API key is available,
    otherwise falls back to TF-IDF cosine similarity.

    Args:
        candidates:      CandidateProfiles from Layer 4.
        hiring_profile:  HiringProfile from Layer 2.
        hyde_result:     HyDEResult from Layer 3 (three ideal profiles).
        force_fallback:  Always use TF-IDF (for testing / offline use).

    Returns:
        dict mapping candidate_id → dense similarity score (0–1).
    """
    if not candidates:
        return {}

    jd_query   = _jd_to_query_text(hiring_profile)
    hyde_texts = hyde_result.all_profile_texts

    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))

    if force_fallback or not has_key:
        logger.info("Dense retrieval running in TF-IDF fallback mode")
        return _score_dense_fallback(candidates, jd_query, hyde_texts)

    try:
        return _score_dense_openai(candidates, jd_query, hyde_texts)
    except Exception as exc:
        logger.warning("OpenAI embedding failed (%s) — falling back to TF-IDF.", exc)
        return _score_dense_fallback(candidates, jd_query, hyde_texts)
