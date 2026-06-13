"""Resume Claim Verification Agent — Layer 5."""

from .agent import verify_candidate, verify_all_candidates
from .schemas import (
    EvidenceItem,
    EvidenceSource,
    VerificationReport,
    VerificationStatus,
    VerifiedClaim,
)
from .utils import merge_external_evidence, status_from_confidence

__all__ = [
    "verify_candidate",
    "verify_all_candidates",
    "VerificationReport",
    "VerifiedClaim",
    "EvidenceItem",
    "EvidenceSource",
    "VerificationStatus",
    "merge_external_evidence",
    "status_from_confidence",
]
