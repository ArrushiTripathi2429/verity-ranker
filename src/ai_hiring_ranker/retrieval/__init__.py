"""
Layer 8 — Hybrid Candidate Retrieval.

Combines four retrieval signals via Reciprocal Rank Fusion (RRF) to
produce a ranked shortlist of candidates before multi-agent evaluation.

Signals:
  1. BM25        — exact keyword / tool name matches (rank-bm25 or TF-IDF fallback)
  2. Dense       — semantic embedding similarity (OpenAI or TF-IDF fallback)
  3. Graph       — adjacent / transferable skill expansion via Layer 7 SkillGraph
  4. HyDE        — similarity to the three ideal candidate profiles from Layer 3

Public API
----------
from ai_hiring_ranker.retrieval import (
    retrieve,           # main entry point → ShortlistResult
    ShortlistResult,
    RetrievalScore,
)
"""

from .retriever import retrieve
from .schemas import RetrievalScore, ShortlistResult

__all__ = [
    "retrieve",
    "ShortlistResult",
    "RetrievalScore",
]
