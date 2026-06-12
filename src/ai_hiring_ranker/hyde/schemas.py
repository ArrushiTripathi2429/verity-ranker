"""
Output schemas for Layer 3 — HyDE Ideal Candidate Generation.

HyDE = Hypothetical Document Embeddings.

Instead of embedding the JD and searching for similar resumes directly,
we generate *hypothetical ideal candidate profiles* first, then embed
those profiles and search against them.  This dramatically improves
retrieval recall because we're searching in "resume space" rather than
"JD space".

Three tiers are always generated:
  - Minimum  : just clears the bar — all required skills, nothing extra
  - Strong   : clearly qualified — required + preferred + evidence depth
  - Exceptional: rare top candidate — all skills + leadership + measurable impact
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CandidateTier(str, Enum):
    MINIMUM = "minimum"
    STRONG = "strong"
    EXCEPTIONAL = "exceptional"


class IdealCandidateProfile(BaseModel):
    """
    A single hypothetical candidate profile at one tier.

    The `profile_text` field is what gets embedded and used for
    semantic retrieval in Layer 8.  It is written in the same style
    as a real resume so the embeddings live in the same vector space.
    """

    tier: CandidateTier = Field(
        ...,
        description="Which tier this profile represents.",
    )
    profile_text: str = Field(
        ...,
        min_length=100,
        description=(
            "A hypothetical resume-style narrative for this tier. "
            "Written in first-person, describes background, skills, projects, "
            "and achievements. Used exclusively for embedding-based retrieval."
        ),
    )
    skills_demonstrated: list[str] = Field(
        default_factory=list,
        description="Skills this profile explicitly demonstrates.",
    )
    experience_years: Optional[int] = Field(
        default=None,
        ge=0,
        description="Approximate years of experience for this tier.",
    )
    seniority_label: str = Field(
        default="",
        description="Human-readable seniority label, e.g. '2–3 years', 'Lead-level'.",
    )
    differentiator: str = Field(
        default="",
        description=(
            "One sentence describing what makes this tier different from the tier below. "
            "Used in audit reports."
        ),
    )

    def word_count(self) -> int:
        return len(self.profile_text.split())


class HyDEResult(BaseModel):
    """
    Container for all three ideal candidate profiles produced by Layer 3.

    Consumed by:
      - Layer 8 (Hybrid Retrieval) — embeds profile_text for dense search
      - Layer 9 (Multi-Agent Evaluation) — uses profiles as calibration anchors
      - Layer 12 (Recruiter Report) — explains what an ideal candidate looks like
    """

    job_title: str = Field(..., description="Job title from the HiringProfile.")
    domain: Optional[str] = Field(default=None)
    profiles: list[IdealCandidateProfile] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly three profiles: minimum, strong, exceptional.",
    )

    def get(self, tier: CandidateTier) -> IdealCandidateProfile:
        """Return the profile for a specific tier."""
        for p in self.profiles:
            if p.tier == tier:
                return p
        raise KeyError(f"Tier {tier} not found in HyDEResult")

    @property
    def minimum(self) -> IdealCandidateProfile:
        return self.get(CandidateTier.MINIMUM)

    @property
    def strong(self) -> IdealCandidateProfile:
        return self.get(CandidateTier.STRONG)

    @property
    def exceptional(self) -> IdealCandidateProfile:
        return self.get(CandidateTier.EXCEPTIONAL)

    @property
    def all_profile_texts(self) -> list[str]:
        """All three profile texts in tier order — ready for batch embedding."""
        return [p.profile_text for p in self.profiles]
