"""
Output schemas for Layer 6 — Evidence Ledger.

The Evidence Ledger is the single auditable source of truth that records
every claim extracted from a candidate's resume together with its
verification outcome from Layer 5. All downstream layers read from this
instead of re-running verification.

Consumed by:
  - Layer 7  (Skill Graph)          — expands verified skills only
  - Layer 8  (Hybrid Retrieval)     — weighs candidates by proof_strength
  - Layer 9  (Multi-Agent Eval)     — Verification Agent reads claim statuses
  - Layer 10 (Rubric Scoring)       — proof_strength dimension
  - Layer 12 (Recruiter Report)     — verified vs unverified claims summary

Design decisions:
  - claim_id is a stable, deterministic hash of (candidate_id + skill + claim_text)
    so the same claim produces the same ID across multiple runs.
  - LedgerEntry mirrors the JSON schema at schemas/evidence_ledger.schema.json
    and adds richer typed fields needed by downstream agents.
  - CandidateLedger aggregates all entries for one candidate and exposes
    the same proof_strength metric used by Layer 5 for consistency.
  - RunLedger wraps all candidates for a single pipeline run, enabling
    one-file serialisation to outputs/.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ..ingestion.schemas import VerificationStatus


# ---------------------------------------------------------------------------
# Claim source — where the primary evidence came from
# ---------------------------------------------------------------------------


class ClaimSource(str, Enum):
    """Primary evidence source that determined the verification status."""

    RESUME    = "resume"    # claim backed only by resume text
    GITHUB    = "github"    # GitHub repo / commit evidence
    KAGGLE    = "kaggle"    # Kaggle competition or notebook evidence
    PORTFOLIO = "portfolio" # personal portfolio / website evidence
    INFERRED  = "inferred"  # logically inferred from another verified claim


# ---------------------------------------------------------------------------
# Single ledger entry — one skill claim with full audit trail
# ---------------------------------------------------------------------------


class LedgerEntry(BaseModel):
    """
    One immutable record in the Evidence Ledger.

    Represents a single skill claim extracted from a resume (Layer 4)
    after it has been processed by the Claim Verification Agent (Layer 5).
    """

    # Identity
    claim_id: str = Field(
        ...,
        description=(
            "Stable, deterministic ID derived from candidate_id + skill + claim_text. "
            "Identical claims across runs share the same claim_id."
        ),
    )
    candidate_id: str = Field(..., description="Candidate this claim belongs to.")

    # Claim content
    skill: str = Field(..., description="Normalised skill name, e.g. 'Python', 'FastAPI'.")
    claim_text: str = Field(
        default="",
        description="The resume sentence(s) that asserted this skill.",
    )

    # Verification outcome
    source: ClaimSource = Field(
        default=ClaimSource.RESUME,
        description="Primary evidence source used to determine verification_status.",
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Final verification label assigned by Layer 5.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Verification confidence score (0–1).",
    )

    # Evidence details
    evidence_url: Optional[str] = Field(
        default=None,
        description="Best URL linking directly to the supporting evidence.",
    )
    evidence_snippet: str = Field(
        default="",
        description="Short excerpt from the evidence (code line, commit message, etc.).",
    )
    recency_years: Optional[float] = Field(
        default=None,
        ge=0,
        description="How many years ago the best evidence was created (0 = this year).",
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of why this status was assigned.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional context, caveats, or flags (e.g. recency penalty applied).",
    )

    # Audit timestamps
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the claim was extracted (Layer 4 timestamp, if available).",
    )
    verified_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the claim was verified (Layer 5 timestamp).",
    )
    ledger_created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this ledger entry was written.",
    )

    # Convenience flags for downstream agents (read-only, derived)
    @property
    def is_proven(self) -> bool:
        """True for VERIFIED or WEAK — has at least some supporting evidence."""
        return self.verification_status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.WEAK,
        )

    @property
    def is_verified(self) -> bool:
        return self.verification_status == VerificationStatus.VERIFIED

    @property
    def is_unsupported(self) -> bool:
        return self.verification_status == VerificationStatus.UNSUPPORTED

    @property
    def is_pending(self) -> bool:
        return self.verification_status == VerificationStatus.PENDING

    @field_validator("skill", mode="before")
    @classmethod
    def normalise_skill(cls, v: str) -> str:
        return v.strip().title()

    def to_export_dict(self) -> dict:
        """Serialise to the format required by schemas/evidence_ledger.schema.json."""
        return {
            "claim_id":            self.claim_id,
            "skill":               self.skill,
            "claim_text":          self.claim_text,
            "source":              self.source.value,
            "verification_status": self.verification_status.value,
            "confidence":          self.confidence,
            "evidence_url":        self.evidence_url,
            "recency_years":       self.recency_years,
            "notes":               self.notes,
        }


# ---------------------------------------------------------------------------
# Candidate ledger — all entries for one candidate
# ---------------------------------------------------------------------------


class CandidateLedger(BaseModel):
    """
    Complete evidence ledger for one candidate.

    Aggregates all LedgerEntries and exposes the same proof_strength
    metric used by VerificationReport (Layer 5) for consistency.
    """

    candidate_id:   str                  = Field(...)
    candidate_name: str                  = Field(default="")
    entries:        list[LedgerEntry]    = Field(default_factory=list)
    run_id:         str                  = Field(
        default="",
        description="Pipeline run ID this ledger was produced in.",
    )
    created_at:     datetime             = Field(default_factory=datetime.utcnow)

    # ---------- aggregate statistics ----------

    @property
    def total_claims(self) -> int:
        return len(self.entries)

    @property
    def verified_count(self) -> int:
        return sum(1 for e in self.entries if e.verification_status == VerificationStatus.VERIFIED)

    @property
    def weak_count(self) -> int:
        return sum(1 for e in self.entries if e.verification_status == VerificationStatus.WEAK)

    @property
    def inferred_count(self) -> int:
        return sum(1 for e in self.entries if e.verification_status == VerificationStatus.INFERRED)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for e in self.entries if e.verification_status == VerificationStatus.UNSUPPORTED)

    @property
    def pending_count(self) -> int:
        return sum(1 for e in self.entries if e.verification_status == VerificationStatus.PENDING)

    @property
    def proof_strength(self) -> float:
        """
        0–1 score representing overall claim verification quality.
        Mirrors the same formula used in VerificationReport.proof_strength
        so scores are directly comparable across layers.

        Weights:
          verified    → 1.0
          weak        → 0.5
          inferred    → 0.25
          unsupported → 0.0
          pending     → 0.1
        """
        if not self.entries:
            return 0.0
        weights = {
            VerificationStatus.VERIFIED:    1.0,
            VerificationStatus.WEAK:        0.5,
            VerificationStatus.INFERRED:    0.25,
            VerificationStatus.UNSUPPORTED: 0.0,
            VerificationStatus.PENDING:     0.1,
        }
        total = sum(weights.get(e.verification_status, 0.0) for e in self.entries)
        return round(total / len(self.entries), 3)

    @property
    def verified_skills(self) -> list[str]:
        return [e.skill for e in self.entries if e.is_verified]

    @property
    def unsupported_skills(self) -> list[str]:
        return [e.skill for e in self.entries if e.is_unsupported]

    @property
    def proven_skills(self) -> list[str]:
        """VERIFIED + WEAK — has some evidence."""
        return [e.skill for e in self.entries if e.is_proven]

    def get_entry(self, skill: str) -> Optional[LedgerEntry]:
        """Return the ledger entry for a specific skill (case-insensitive)."""
        skill_norm = skill.strip().title()
        return next((e for e in self.entries if e.skill == skill_norm), None)

    def summary_line(self) -> str:
        return (
            f"{self.candidate_id}: "
            f"{self.verified_count}✓ verified  "
            f"{self.weak_count}~ weak  "
            f"{self.inferred_count}? inferred  "
            f"{self.unsupported_count}✗ unsupported  "
            f"proof_strength={self.proof_strength:.2f}"
        )

    def to_export_dict(self) -> dict:
        """Serialise to the format required by schemas/evidence_ledger.schema.json."""
        return {
            "candidate_id": self.candidate_id,
            "claims":       [e.to_export_dict() for e in self.entries],
        }


# ---------------------------------------------------------------------------
# Run ledger — all candidates for one pipeline run
# ---------------------------------------------------------------------------


class RunLedger(BaseModel):
    """
    Full ledger for one pipeline run covering all candidates.

    Serialised as a single JSON file to outputs/runs/<run_id>_ledger.json
    so every run is fully auditable and reproducible.
    """

    run_id:      str                     = Field(...)
    job_title:   str                     = Field(default="")
    candidates:  list[CandidateLedger]   = Field(default_factory=list)
    created_at:  datetime                = Field(default_factory=datetime.utcnow)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def total_claims(self) -> int:
        return sum(c.total_claims for c in self.candidates)

    def get_candidate(self, candidate_id: str) -> Optional[CandidateLedger]:
        return next((c for c in self.candidates if c.candidate_id == candidate_id), None)

    def proof_strength_ranking(self) -> list[tuple[str, float]]:
        """Return candidates sorted by proof_strength descending."""
        ranked = sorted(
            self.candidates,
            key=lambda c: c.proof_strength,
            reverse=True,
        )
        return [(c.candidate_id, c.proof_strength) for c in ranked]

    def to_export_dict(self) -> dict:
        return {
            "run_id":          self.run_id,
            "job_title":       self.job_title,
            "created_at":      self.created_at.isoformat(),
            "candidate_count": self.candidate_count,
            "total_claims":    self.total_claims,
            "candidates":      [c.to_export_dict() for c in self.candidates],
        }
