"""
Shared helpers for Layer 5 claim verification.
"""

from __future__ import annotations

from typing import Optional

from ..config import VerificationConfig, VerificationLabelsConfig, get_verification_config
from ..ingestion.schemas import VerificationStatus
from .schemas import EvidenceItem, VerificationReport, VerifiedClaim


def status_from_confidence(
    confidence: float,
    labels: Optional[VerificationLabelsConfig] = None,
) -> VerificationStatus:
    """Map a confidence score to a verification label using config thresholds."""
    cfg = labels or get_verification_config().verification_labels
    if confidence >= cfg.verified.min_confidence:
        return VerificationStatus.VERIFIED
    if confidence >= cfg.weak.min_confidence:
        return VerificationStatus.WEAK
    if confidence >= cfg.inferred.min_confidence:
        return VerificationStatus.INFERRED
    return VerificationStatus.UNSUPPORTED


def apply_recency_penalty(
    evidence: EvidenceItem,
    recency_cutoff_years: float,
) -> EvidenceItem:
    """Down-rank evidence older than the configured cutoff."""
    if evidence.recency_years is None:
        return evidence
    if evidence.recency_years <= recency_cutoff_years:
        return evidence
    penalty = max(0.35, 1.0 - (evidence.recency_years - recency_cutoff_years) * 0.15)
    return evidence.model_copy(
        update={"relevance_score": round(evidence.relevance_score * penalty, 3)}
    )


def evidence_matches_skill(evidence: EvidenceItem, skill: str) -> bool:
    """Return True if an evidence item supports the given skill claim."""
    skill_lower = skill.lower()
    if evidence.skill and evidence.skill.lower() == skill_lower:
        return True
    snippet = evidence.snippet.lower()
    return skill_lower in snippet


def merge_evidence_into_claim(
    claim: VerifiedClaim,
    new_evidence: list[EvidenceItem],
    *,
    confidence_boost: float = 0.20,
    labels: Optional[VerificationLabelsConfig] = None,
) -> VerifiedClaim:
    """Blend external evidence into an existing verified claim."""
    if not new_evidence:
        return claim

    combined = claim.evidence + new_evidence
    best = max(new_evidence, key=lambda e: e.relevance_score)
    new_confidence = min(
        round(claim.confidence + confidence_boost * best.relevance_score, 3),
        1.0,
    )
    new_status = status_from_confidence(new_confidence, labels)
    reasoning = (
        f"{claim.reasoning}; external evidence found "
        f"({len(new_evidence)} items, best relevance={best.relevance_score:.2f})"
    )
    return claim.model_copy(
        update={
            "status": new_status,
            "confidence": new_confidence,
            "evidence": combined,
            "reasoning": reasoning,
        }
    )


def merge_external_evidence(
    report: VerificationReport,
    external_evidence: list[EvidenceItem],
    *,
    confidence_boost: float = 0.20,
    config: Optional[VerificationConfig] = None,
) -> VerificationReport:
    """Merge skill-tagged external evidence into a verification report."""
    cfg = config or get_verification_config()
    updated_claims: list[VerifiedClaim] = []

    for claim in report.claims:
        matched = [e for e in external_evidence if evidence_matches_skill(e, claim.skill)]
        updated_claims.append(
            merge_evidence_into_claim(
                claim,
                matched,
                confidence_boost=confidence_boost,
                labels=cfg.verification_labels,
            )
        )

    return report.model_copy(update={"claims": updated_claims})
