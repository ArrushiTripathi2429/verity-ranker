"""
Output schemas for Layer 9 — Multi-Agent Evaluation.

Each specialist agent evaluates one dimension of a candidate and returns
an AgentVerdict. The orchestrator collects all verdicts into an
EvaluationResult per candidate.

Five agents, each with a clear role:
  1. JD Fit Agent          — how well the candidate matches the JD
  2. Technical Fit Agent   — depth and breadth of technical skills
  3. Career Trajectory Agent — growth signal and seniority fit
  4. Verification Agent    — how much of the profile is evidence-backed
  5. Final Ranking Agent   — synthesises all verdicts into a unified score

Consumed by:
  - Layer 10 (Rubric Scoring)  — uses per-dimension scores
  - Layer 11 (Listwise Rerank) — uses EvaluationResult for global comparison
  - Layer 12 (Recruiter Report)— uses reasoning + flags for candidate cards
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------


class AgentRole(str, Enum):
    JD_FIT           = "jd_fit"
    TECHNICAL_FIT    = "technical_fit"
    TRAJECTORY       = "trajectory"
    VERIFICATION     = "verification"
    FINAL_SYNTHESIS  = "final_synthesis"


# ---------------------------------------------------------------------------
# Single agent verdict for one candidate
# ---------------------------------------------------------------------------


class AgentVerdict(BaseModel):
    """
    The verdict from one specialist agent for one candidate.

    score     — 0–1 normalised dimension score.
    reasoning — 2–5 sentences explaining the score. Always evidence-cited.
    flags     — list of concerns or notable positives (short phrases).
    evidence  — specific resume snippets or URLs that support the reasoning.
    """

    agent:         AgentRole        = Field(...)
    candidate_id:  str              = Field(...)
    score:         float            = Field(ge=0.0, le=1.0)
    reasoning:     str              = Field(default="")
    flags:         list[str]        = Field(default_factory=list)
    evidence:      list[str]        = Field(
        default_factory=list,
        description="Resume snippets or URLs cited in the reasoning.",
    )
    evaluated_at:  datetime         = Field(default_factory=datetime.utcnow)

    @property
    def score_label(self) -> str:
        if self.score >= 0.80:
            return "strong"
        if self.score >= 0.55:
            return "moderate"
        if self.score >= 0.30:
            return "weak"
        return "poor"


# ---------------------------------------------------------------------------
# Per-dimension scores — structured output of the final synthesis agent
# ---------------------------------------------------------------------------


class DimensionScores(BaseModel):
    """
    Six rubric dimension scores (0–1) for one candidate.
    Mirrors the six weights in configs/v2/scoring_weights.yaml.
    These feed directly into Layer 10 (Rubric Scoring).
    """

    skill_fit:          float = Field(default=0.0, ge=0.0, le=1.0)
    experience_depth:   float = Field(default=0.0, ge=0.0, le=1.0)
    seniority_match:    float = Field(default=0.0, ge=0.0, le=1.0)
    domain_match:       float = Field(default=0.0, ge=0.0, le=1.0)
    career_growth:      float = Field(default=0.0, ge=0.0, le=1.0)
    proof_strength:     float = Field(default=0.0, ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "skill_fit":        self.skill_fit,
            "experience_depth": self.experience_depth,
            "seniority_match":  self.seniority_match,
            "domain_match":     self.domain_match,
            "career_growth":    self.career_growth,
            "proof_strength":   self.proof_strength,
        }


# ---------------------------------------------------------------------------
# Full evaluation result for one candidate
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """
    Complete multi-agent evaluation result for one candidate.

    Aggregates all five AgentVerdicts and the final DimensionScores.
    This is the primary input to Layer 10 (Rubric Scoring) and
    Layer 11 (Listwise Re-Ranking).
    """

    candidate_id:    str                  = Field(...)
    candidate_name:  str                  = Field(default="")
    verdicts:        list[AgentVerdict]   = Field(default_factory=list)
    dimensions:      DimensionScores      = Field(default_factory=DimensionScores)

    # Synthesised outputs from the Final Ranking Agent
    strengths:       list[str]            = Field(
        default_factory=list,
        description="Top 3–5 candidate strengths, evidence-cited.",
    )
    risks:           list[str]            = Field(
        default_factory=list,
        description="Top 1–3 risks or gaps identified.",
    )
    summary:         str                  = Field(
        default="",
        description="2–3 sentence overall assessment.",
    )
    evaluated_at:    datetime             = Field(default_factory=datetime.utcnow)

    # Convenience
    def get_verdict(self, role: AgentRole) -> Optional[AgentVerdict]:
        return next((v for v in self.verdicts if v.agent == role), None)

    @property
    def overall_score(self) -> float:
        """Simple average of the six dimension scores — used for quick sorting."""
        d = self.dimensions
        vals = [
            d.skill_fit, d.experience_depth, d.seniority_match,
            d.domain_match, d.career_growth, d.proof_strength,
        ]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def verdict_summary(self) -> str:
        lines = [f"{self.candidate_id}  overall={self.overall_score:.2f}"]
        for v in self.verdicts:
            lines.append(f"  [{v.agent.value:18s}] score={v.score:.2f}  {v.score_label}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch evaluation result — all candidates in the shortlist
# ---------------------------------------------------------------------------


class BatchEvaluationResult(BaseModel):
    """
    All EvaluationResults for a pipeline run's shortlist.
    Sorted by overall_score descending.
    """

    job_title:   str                      = Field(default="")
    results:     list[EvaluationResult]   = Field(default_factory=list)
    evaluated_at: datetime                = Field(default_factory=datetime.utcnow)

    @property
    def ranked(self) -> list[EvaluationResult]:
        return sorted(self.results, key=lambda r: r.overall_score, reverse=True)

    def get(self, candidate_id: str) -> Optional[EvaluationResult]:
        return next((r for r in self.results if r.candidate_id == candidate_id), None)

    def summary_table(self) -> str:
        lines = [f"{'Rank':<5} {'Candidate':<12} {'Overall':>7}  Dim scores"]
        for rank, r in enumerate(self.ranked, 1):
            d = r.dimensions
            lines.append(
                f"{rank:<5} {r.candidate_id:<12} {r.overall_score:>7.3f}  "
                f"skl={d.skill_fit:.2f} exp={d.experience_depth:.2f} "
                f"sen={d.seniority_match:.2f} dom={d.domain_match:.2f} "
                f"trj={d.career_growth:.2f} prf={d.proof_strength:.2f}"
            )
        return "\n".join(lines)
