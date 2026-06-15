"""
Reciprocal Rank Fusion (RRF) for Layer 8.

RRF combines multiple ranked lists into a single unified ranking without
needing to tune score weights or normalise across incomparable scales.

Formula (Cormack et al., 2009):
    RRF(d) = Σ_r  1 / (k + rank_r(d))

where:
    rank_r(d) = position of document d in ranked list r (1-based)
    k         = smoothing constant (default 60; higher = reduces impact
                of very high rankings)

Why RRF instead of weighted sum?
  - BM25 scores (0–∞) and cosine scores (0–1) are on different scales.
    Normalising them introduces its own bias.
  - RRF only uses *rank positions*, not raw scores — so it's robust to
    score scale differences and outlier documents.
  - Adding or removing a retrieval signal only requires adding/removing
    a ranked list, not re-tuning weights.

Public API
----------
reciprocal_rank_fusion(ranked_lists, k=60) → list[tuple[str, float]]
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Core RRF implementation
# ---------------------------------------------------------------------------


def _rank_list(scores: dict[str, float]) -> list[str]:
    """Sort candidate IDs by score descending → rank-ordered list."""
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def reciprocal_rank_fusion(
    ranked_lists: list[dict[str, float]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """
    Fuse multiple score dicts into one ranked list using Reciprocal Rank Fusion.

    Args:
        ranked_lists:  One dict per retrieval signal, mapping
                       candidate_id → signal score (used only for ranking order).
        k:             RRF smoothing constant (default 60 — standard literature value).
        weights:       Optional per-signal weights to amplify certain signals.
                       Must be same length as ranked_lists. Default: equal weights.

    Returns:
        List of (candidate_id, rrf_score) sorted by rrf_score descending.

    Example:
        >>> fusion = reciprocal_rank_fusion([bm25_scores, dense_scores, graph_scores])
        >>> top5 = fusion[:5]
    """
    if not ranked_lists:
        return []

    # Default equal weights
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match "
            f"ranked_lists length ({len(ranked_lists)})"
        )

    # Collect all candidate IDs across all lists
    all_candidates: set[str] = set()
    for score_dict in ranked_lists:
        all_candidates.update(score_dict.keys())

    rrf_scores: dict[str, float] = {}

    for signal_scores, w in zip(ranked_lists, weights):
        ranked = _rank_list(signal_scores)
        rank_map = {cid: rank + 1 for rank, cid in enumerate(ranked)}

        for cid in all_candidates:
            rank = rank_map.get(cid, len(ranked) + 1)  # unseen → last rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w * (1.0 / (k + rank))

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def build_rrf_notes(
    candidate_id: str,
    bm25_scores:  dict[str, float],
    dense_scores: dict[str, float],
    graph_scores: dict[str, float],
    hyde_scores:  dict[str, float],
) -> str:
    """
    Build a human-readable explanation of why a candidate ranked where they did.

    Used for the audit trail in RetrievalScore.retrieval_notes.
    """
    bm25  = bm25_scores.get(candidate_id, 0.0)
    dense = dense_scores.get(candidate_id, 0.0)
    graph = graph_scores.get(candidate_id, 0.0)
    hyde  = hyde_scores.get(candidate_id, 0.0)

    signals = [
        ("BM25 (keyword match)", bm25),
        ("dense semantic similarity", dense),
        ("skill graph expansion", graph),
        ("HyDE profile similarity", hyde),
    ]
    top_signal = max(signals, key=lambda x: x[1])

    parts: list[str] = []

    if top_signal[1] >= 0.7:
        parts.append(f"Strong {top_signal[0]} score ({top_signal[1]:.2f}).")
    elif top_signal[1] >= 0.4:
        parts.append(f"Moderate {top_signal[0]} score ({top_signal[1]:.2f}).")
    else:
        parts.append("Low scores across all signals.")

    # Note weak signals
    for name, score in signals:
        if score < 0.2 and name != top_signal[0]:
            parts.append(f"Weak {name} ({score:.2f}).")

    return " ".join(parts)
