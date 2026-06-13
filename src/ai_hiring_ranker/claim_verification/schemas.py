"""
Output schemas for Layer 5 — Resume Claim Verification Agent.

Every skill claim from Layer 4 gets a VerifiedClaim after this layer.
The VerificationReport for a candidate is consumed by:
  - Layer 6  (Evidence Ledger)   — stores claims with proof as an audit trail
  - Layer 9  (Multi-Agent Eval)  — Verification Agent reads proof_strength
  - Layer 10 (Rubric Scoring)    — proof_strength dimension
  - Layer 12 (Recruiter Report)  — verified vs unverified claims summary

Verification label definitions (from verification_rules.yaml):
  VERIFIED    — direct code/commit evidence; recent and relevant      (conf ≥ 0.75)
  WEAK        — indirect, old, or low-volume evidence                 (conf ≥ 0.40)
  INFERRED    — adjacent evidence (FastAPI repo → implies Python)     (conf ≥ 0.20)
  UNSUPPORTED — no evidence found for the claim                       (conf = 0.0)
  PENDING     — not yet verified (GitHub unavailable / no link)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# Re-export so callers don't need to import from ingestion.schemas
from ..ingestion.schemas import VerificationStatus


# ---------------------------------------------------------------------------
# Evidence source
# ---------------------------------------------------------------------------


class EvidenceSource(str, Enum):
    RESUME       = "resume"      # claim comes from resume text only
    GITHUB_REPO  = "github_repo" # found in a GitHub repository
    GITHUB_COMMIT= "github_commit"
    GITHUB_FILE  = "github_file"
    KAGGLE       = "kaggle"
    PORTFOLIO    = "portfolio"
    INFERRED     = "inferred"    # logically inferred from another verified claim


# ---------------------------------------------------------------------------
# Single evidence item
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """One piece of evidence supporting (or contradicting) a skill claim."""

    source:       EvidenceSource = Field(..., description="Where this evidence came from.")
    skill:        Optional[str]  = Field(
        default=None,
        description="Normalised skill name this evidence supports, if known.",
    )
    url:          Optional[str]  = Field(default=None, description="Direct URL to the evidence.")
    snippet:      str            = Field(default="", description="Relevant text or code excerpt.")
    file_path:    Optional[str]  = Field(default=None, description="File path within a repo, if applicable.")
    commit_sha:   Optional[str]  = Field(default=None, description="Commit SHA if from a commit.")
    recency_years: Optional[float] = Field(
        default=None, ge=0,
        description="How many years ago this evidence was created (0 = very recent).",
    )
    relevance_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="How directly this evidence supports the claim (0–1).",
    )


# ---------------------------------------------------------------------------
# Verified claim
# ---------------------------------------------------------------------------


class VerifiedClaim(BaseModel):
    """
    A single skill claim with its verification result.

    One VerifiedClaim per SkillClaim from Layer 4.
    """

    candidate_id:  str               = Field(..., description="Candidate this claim belongs to.")
    skill:         str               = Field(..., description="Normalised skill name.")
    claim_text:    str               = Field(
        default="",
        description="The resume sentence(s) that made this claim.",
    )
    status:        VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Verification outcome label.",
    )
    confidence:    float             = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Verification confidence score (0–1).",
    )
    evidence:      list[EvidenceItem] = Field(
        default_factory=list,
        description="All evidence items collected for this claim.",
    )
    reasoning:     str               = Field(
        default="",
        description="Human-readable explanation of why this status was assigned.",
    )
    verified_at:   datetime          = Field(default_factory=datetime.utcnow)

    @field_validator("skill", mode="before")
    @classmethod
    def normalise_skill(cls, v: str) -> str:
        return v.strip().title()

    # Convenience
    @property
    def is_verified(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    @property
    def is_supported(self) -> bool:
        return self.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.WEAK,
            VerificationStatus.INFERRED,
        )

    @property
    def best_evidence_url(self) -> Optional[str]:
        urls = [e.url for e in self.evidence if e.url]
        return urls[0] if urls else None


# ---------------------------------------------------------------------------
# Full verification report for one candidate
# ---------------------------------------------------------------------------


class VerificationReport(BaseModel):
    """
    Complete verification report for one candidate.
    Aggregates all VerifiedClaims and computes summary statistics.
    """

    candidate_id:   str                  = Field(...)
    candidate_name: str                  = Field(default="")
    github_url:     Optional[str]        = Field(default=None)
    kaggle_url:     Optional[str]        = Field(default=None)
    claims:         list[VerifiedClaim]  = Field(default_factory=list)
    github_checked: bool                 = Field(default=False)
    verified_at:    datetime             = Field(default_factory=datetime.utcnow)
    error_notes:    list[str]            = Field(
        default_factory=list,
        description="Non-fatal errors encountered during verification (e.g. API rate limit).",
    )

    # ---------- aggregate statistics ----------

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def verified_count(self) -> int:
        return sum(1 for c in self.claims if c.status == VerificationStatus.VERIFIED)

    @property
    def weak_count(self) -> int:
        return sum(1 for c in self.claims if c.status == VerificationStatus.WEAK)

    @property
    def inferred_count(self) -> int:
        return sum(1 for c in self.claims if c.status == VerificationStatus.INFERRED)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for c in self.claims if c.status == VerificationStatus.UNSUPPORTED)

    @property
    def pending_count(self) -> int:
        return sum(1 for c in self.claims if c.status == VerificationStatus.PENDING)

    @property
    def proof_strength(self) -> float:
        """
        0–1 score representing overall claim verification quality.

        Weights:
          verified   → 1.0
          weak       → 0.5
          inferred   → 0.25
          unsupported → 0.0
          pending    → 0.1
        """
        if not self.claims:
            return 0.0
        weights = {
            VerificationStatus.VERIFIED:    1.0,
            VerificationStatus.WEAK:        0.5,
            VerificationStatus.INFERRED:    0.25,
            VerificationStatus.UNSUPPORTED: 0.0,
            VerificationStatus.PENDING:     0.1,
        }
        total = sum(weights.get(c.status, 0.0) for c in self.claims)
        return round(total / len(self.claims), 3)

    @property
    def verified_skill_names(self) -> list[str]:
        return [c.skill for c in self.claims if c.status == VerificationStatus.VERIFIED]

    @property
    def unsupported_skill_names(self) -> list[str]:
        return [c.skill for c in self.claims if c.status == VerificationStatus.UNSUPPORTED]

    def get_claim(self, skill: str) -> Optional[VerifiedClaim]:
        skill_norm = skill.strip().title()
        return next((c for c in self.claims if c.skill == skill_norm), None)

    def summary_line(self) -> str:
        return (
            f"{self.candidate_id}: {self.verified_count}✓ verified  "
            f"{self.weak_count}~ weak  "
            f"{self.unsupported_count}✗ unsupported  "
            f"proof_strength={self.proof_strength:.2f}"
        )

    def to_export_dict(self) -> dict:
        """Serialize for JSON output / submission artifacts."""
        return self.model_dump(mode="json")
