"""
Output schemas for Layer 8 — Hybrid Candidate Retrieval.

The retrieval layer takes the HiringProfile (Layer 2), HyDE profiles
(Layer 3), and all CandidateProfiles (Layer 4), and returns a ranked
shortlist of the most relevant candidates before full multi-agent
evaluation begins.

Four retrieval signals are combined via Reciprocal Rank Fusion (RRF):
  1. BM25        — exact keyword / tool name matches
  2. Dense       — semantic embedding similarity
  3. Graph       — adjacent / transferable skill expansion (Layer 7)
  4. HyDE        — similarity to the three ideal candidate profiles

Consumed by:
  - Layer 9  (Multi-Agent Evaluation) — receives the shortlisted candidates
  - Layer 10 (Rubric Scoring)         — scores only shortlisted candidates
  - Layer 12 (Recruiter Report)       — explains why candidates were shortlisted

Design notes:
  - RetrievalScore stores all four raw signal scores for full auditability.
    The recruiter report can explain "why this candidate was shortlisted".
  - ShortlistResult sorts by rrf_score and exposes a .top_k() method so
    callers don't need to re-sort.
  - The schemas are deliberately decoupled from the retrieval engine so
    the engine can be swapped (e.g. FAISS → Qdrant) without changing
    any downstream layer.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Per-signal scores for one candidate
# ---------------------------------------------------------------------------


class RetrievalScore(BaseModel):
    """
    All retrieval signal scores for one candidate.

    Every score is 0–1 normalised. The rrf_score is the final fused
    ranking score computed via Reciprocal Rank Fusion.
    """

    candidate_id: str = Field(..., description="Candidate identifier.")

    # Raw signal scores (0–1)
    bm25_score:   float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="BM25 lexical match score against the JD skill terms.",
    )
    dense_score:  float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between candidate embedding and JD/HyDE embeddings.",
    )
    graph_score:  float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Skill graph expansion match score (Layer 7).",
    )
    hyde_score:   float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity to the three HyDE ideal candidate profiles.",
    )

    # Fused score
    rrf_score:    float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Reciprocal Rank Fusion score — the primary sort key. "
            "Higher is better. Not capped at 1.0 (RRF sums reciprocal ranks)."
        ),
    )

    # Matched skills for audit trail
    matched_required:   list[str] = Field(
        default_factory=list,
        description="Required JD skills found in this candidate's profile.",
    )
    matched_preferred:  list[str] = Field(
        default_factory=list,
        description="Preferred JD skills found in this candidate's profile.",
    )
    graph_expanded:     list[str] = Field(
        default_factory=list,
        description="Skills matched via graph expansion (adjacent/transferable).",
    )

    # Shortlist metadata
    shortlist_rank:     Optional[int]  = Field(
        default=None,
        description="1-based rank within the shortlist (set after RRF fusion).",
    )
    retrieval_notes:    str = Field(
        default="",
        description="Human-readable explanation of why this candidate was shortlisted.",
    )

    @property
    def required_match_ratio(self) -> float:
        """Proportion of required skills matched (0–1). Requires total_required to be set."""
        return 0.0  # computed externally; placeholder for type completeness

    @property
    def dominant_signal(self) -> str:
        """Which of the four signals contributed most to this candidate's score."""
        signals = {
            "bm25":  self.bm25_score,
            "dense": self.dense_score,
            "graph": self.graph_score,
            "hyde":  self.hyde_score,
        }
        return max(signals, key=lambda k: signals[k])

    def score_breakdown(self) -> str:
        return (
            f"bm25={self.bm25_score:.3f}  "
            f"dense={self.dense_score:.3f}  "
            f"graph={self.graph_score:.3f}  "
            f"hyde={self.hyde_score:.3f}  "
            f"rrf={self.rrf_score:.4f}"
        )


# ---------------------------------------------------------------------------
# Full shortlist result
# ---------------------------------------------------------------------------


class ShortlistResult(BaseModel):
    """
    Ranked shortlist of candidates produced by the hybrid retrieval layer.

    The `scores` list is sorted by rrf_score descending on construction.
    Use .top_k(n) to get the top-n candidates for multi-agent evaluation.
    """

    job_title:          str                 = Field(default="")
    total_candidates:   int                 = Field(
        default=0,
        description="Total number of candidates considered before shortlisting.",
    )
    scores:             list[RetrievalScore] = Field(
        default_factory=list,
        description="All scored candidates, sorted by rrf_score descending.",
    )
    required_skills:    list[str]           = Field(
        default_factory=list,
        description="Required skills from the JD (used for match ratio reporting).",
    )
    preferred_skills:   list[str]           = Field(
        default_factory=list,
        description="Preferred skills from the JD.",
    )
    retrieval_config:   dict                = Field(
        default_factory=dict,
        description="Config snapshot: k, weights, max_hops used in this run.",
    )

    @field_validator("scores", mode="before")
    @classmethod
    def sort_by_rrf(cls, v: list) -> list:
        """Always store scores sorted by rrf_score descending."""
        if isinstance(v, list):
            return sorted(v, key=lambda s: s.rrf_score if hasattr(s, "rrf_score") else s.get("rrf_score", 0), reverse=True)
        return v

    def top_k(self, k: int) -> list[RetrievalScore]:
        """Return the top-k candidates by RRF score."""
        return self.scores[:k]

    @property
    def shortlisted_ids(self) -> list[str]:
        return [s.candidate_id for s in self.scores]

    def get_score(self, candidate_id: str) -> Optional[RetrievalScore]:
        return next((s for s in self.scores if s.candidate_id == candidate_id), None)

    def summary(self) -> str:
        return (
            f"Shortlist: {len(self.scores)}/{self.total_candidates} candidates  "
            f"| top: {self.scores[0].candidate_id if self.scores else 'none'}  "
            f"| rrf={self.scores[0].rrf_score:.4f}" if self.scores else "empty shortlist"
        )
