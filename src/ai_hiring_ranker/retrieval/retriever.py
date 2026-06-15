"""
Hybrid Candidate Retrieval — Layer 8.

Orchestrates all four retrieval signals and fuses them via RRF to produce
a ranked shortlist of candidates for multi-agent evaluation (Layer 9).

Pipeline inside this layer:
    CandidateProfiles + HiringProfile + HyDEResult
      ↓  BM25          — exact keyword / tool name matches
      ↓  Dense         — semantic embedding similarity (JD + HyDE vectors)
      ↓  Graph         — adjacent / transferable skill expansion (Layer 7)
      ↓  RRF Fusion    — combines all four signal rankings
      ↓  Shortlist     — top-k by RRF score, with full audit trail
    ShortlistResult

Public API
----------
retrieve(candidates, hiring_profile, hyde_result, k=25, ...)
    → ShortlistResult
"""

from __future__ import annotations

import logging
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile
from ..hyde.schemas import HyDEResult
from ..jd_intelligence.schemas import HiringProfile
from .bm25_retriever import score_bm25
from .dense_retriever import score_dense
from .graph_retriever import get_matched_skills, score_graph
from .rrf import build_rrf_notes, reciprocal_rank_fusion
from .schemas import RetrievalScore, ShortlistResult

logger = logging.getLogger(__name__)

# Default RRF signal weights.
# BM25 and graph are boosted slightly because:
#   - BM25 catches exact tool/library names that embeddings miss
#   - Graph expansion gives partial credit for adjacent skills (fairer)
# These are sensible defaults; callers can override via signal_weights.
_DEFAULT_WEIGHTS = {
    "bm25":  1.20,
    "dense": 1.00,
    "graph": 1.10,
    "hyde":  1.00,
}


def retrieve(
    candidates: list[CandidateProfile],
    hiring_profile: HiringProfile,
    hyde_result: HyDEResult,
    *,
    k: int = 25,
    force_fallback: bool = False,
    signal_weights: Optional[dict[str, float]] = None,
) -> ShortlistResult:
    """
    Run hybrid retrieval and return a ranked shortlist.

    Args:
        candidates:      All CandidateProfiles from Layer 4.
        hiring_profile:  HiringProfile from Layer 2.
        hyde_result:     HyDEResult from Layer 3.
        k:               Maximum shortlist size (default 25).
                         If fewer candidates exist, all are returned.
        force_fallback:  Use TF-IDF fallback instead of OpenAI embeddings.
        signal_weights:  Override the default RRF signal weights.
                         Keys: 'bm25', 'dense', 'graph', 'hyde'.

    Returns:
        ShortlistResult with candidates sorted by RRF score.
    """
    if not candidates:
        logger.warning("retrieve() called with empty candidate list.")
        return ShortlistResult(
            job_title=hiring_profile.job_title,
            total_candidates=0,
            required_skills=hiring_profile.all_required_skill_names,
            preferred_skills=hiring_profile.all_preferred_skill_names,
        )

    weights = {**_DEFAULT_WEIGHTS, **(signal_weights or {})}
    logger.info(
        "Layer 8 retrieval: %d candidates, k=%d, weights=%s",
        len(candidates),
        k,
        weights,
    )

    # ── Step 1: Collect all JD terms for BM25 ──────────────────────────
    jd_terms = (
        hiring_profile.all_required_skill_names
        + hiring_profile.all_preferred_skill_names
        + hiring_profile.key_responsibilities[:5]
    )

    # ── Step 2: Run all four retrieval signals ──────────────────────────
    logger.info("Running BM25...")
    bm25_scores  = score_bm25(candidates, jd_terms)

    logger.info("Running dense retrieval...")
    dense_scores = score_dense(
        candidates, hiring_profile, hyde_result,
        force_fallback=force_fallback,
    )

    logger.info("Running graph-expansion retrieval...")
    graph_scores = score_graph(candidates, hiring_profile)

    # HyDE score — dense retrieval already encodes HyDE similarity, but we
    # keep it as a separate signal using the HyDE-only portion of the dense
    # model for better RRF signal separation.
    # Here we reuse the dense score as a proxy when no separate HyDE embedder
    # exists. When a dedicated embedding engine is available, replace this.
    hyde_scores: dict[str, float] = {
        c.candidate_id: round(dense_scores.get(c.candidate_id, 0.0) * 0.9, 4)
        for c in candidates
    }

    # ── Step 3: RRF fusion ──────────────────────────────────────────────
    logger.info("Fusing signals via RRF...")
    fused = reciprocal_rank_fusion(
        [bm25_scores, dense_scores, graph_scores, hyde_scores],
        k=60,
        weights=[
            weights["bm25"],
            weights["dense"],
            weights["graph"],
            weights["hyde"],
        ],
    )

    # ── Step 4: Build per-candidate RetrievalScore objects ──────────────
    scores: list[RetrievalScore] = []
    for rank, (candidate_id, rrf_score) in enumerate(fused, start=1):
        # Find the matching CandidateProfile for skill audit trail
        profile = next((c for c in candidates if c.candidate_id == candidate_id), None)

        matched_req: list[str] = []
        matched_pref: list[str] = []
        graph_exp: list[str] = []

        if profile is not None:
            matched_req, matched_pref, graph_exp = get_matched_skills(
                profile, hiring_profile
            )

        notes = build_rrf_notes(
            candidate_id,
            bm25_scores,
            dense_scores,
            graph_scores,
            hyde_scores,
        )

        scores.append(
            RetrievalScore(
                candidate_id=candidate_id,
                bm25_score=bm25_scores.get(candidate_id, 0.0),
                dense_score=dense_scores.get(candidate_id, 0.0),
                graph_score=graph_scores.get(candidate_id, 0.0),
                hyde_score=hyde_scores.get(candidate_id, 0.0),
                rrf_score=round(rrf_score, 6),
                matched_required=matched_req,
                matched_preferred=matched_pref,
                graph_expanded=graph_exp,
                shortlist_rank=rank,
                retrieval_notes=notes,
            )
        )

    # ── Step 5: Trim to top-k ───────────────────────────────────────────
    shortlisted = scores[:k]

    logger.info(
        "Shortlist: %d/%d candidates  |  top: %s (rrf=%.4f)",
        len(shortlisted),
        len(candidates),
        shortlisted[0].candidate_id if shortlisted else "none",
        shortlisted[0].rrf_score if shortlisted else 0.0,
    )

    return ShortlistResult(
        job_title=hiring_profile.job_title,
        total_candidates=len(candidates),
        scores=shortlisted,
        required_skills=hiring_profile.all_required_skill_names,
        preferred_skills=hiring_profile.all_preferred_skill_names,
        retrieval_config={
            "k": k,
            "signal_weights": weights,
            "force_fallback": force_fallback,
        },
    )
