"""HyDE Ideal Candidate Generation — Layer 3."""

from .generator import generate_hyde_profiles
from .schemas import CandidateTier, HyDEResult, IdealCandidateProfile

__all__ = [
    "generate_hyde_profiles",
    "HyDEResult",
    "IdealCandidateProfile",
    "CandidateTier",
]
