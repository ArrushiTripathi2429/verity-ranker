"""
Output schemas for Layer 4 — Candidate Profile Extraction.

A CandidateProfile is the structured, typed representation of a resume
that every downstream layer works with. Nobody reads raw_text after this.

Key design decisions:
- Skills carry evidence snippets so Layer 5 (Verification) has context to check.
- Career timeline is a list of ordered roles — enables trajectory analysis in Layer 9.
- Achievements are separated from responsibilities — "built X" vs "responsible for X".
- Seniority and career signals are numeric (0-1) so Layer 10 can use them directly.
- Every extracted field stores the source sentence so audits are traceable.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator





class SkillConfidence(str, Enum):
    """How confidently was this skill extracted from the resume?"""
    EXPLICIT   = "explicit"    # listed in a Skills section or named directly
    INFERRED   = "inferred"    # implied by a project, tool, or context
    WEAK       = "weak"        # mentioned once, no supporting evidence


class EmploymentCategory(str, Enum):
    FULL_TIME  = "full_time"
    PART_TIME  = "part_time"
    INTERNSHIP = "internship"
    CONTRACT   = "contract"
    FREELANCE  = "freelance"
    ACADEMIC   = "academic"
    UNKNOWN    = "unknown"


class DegreeLevel(str, Enum):
    PHD        = "phd"
    MASTERS    = "masters"
    BACHELORS  = "bachelors"
    ASSOCIATE  = "associate"
    BOOTCAMP   = "bootcamp"
    SELF_TAUGHT = "self_taught"
    UNKNOWN    = "unknown"




class SkillClaim(BaseModel):
    """A single skill extracted from the resume with its evidence context."""

    skill: str = Field(
        ...,
        description="Normalised skill name, e.g. 'Python', 'FastAPI'.",
    )
    confidence: SkillConfidence = Field(
        default=SkillConfidence.EXPLICIT,
        description="How confidently was this skill extracted.",
    )
    evidence_snippets: list[str] = Field(
        default_factory=list,
        description=(
            "1–3 sentences from the resume that demonstrate this skill. "
            "Used by Layer 5 (Claim Verification) as the primary evidence source."
        ),
    )
    years_of_experience: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated years of experience with this skill, if extractable.",
    )
    last_used_year: Optional[int] = Field(
        default=None,
        description="Most recent year this skill was used, if detectable.",
    )

    @field_validator("skill", mode="before")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().title()


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


class ProjectEntry(BaseModel):
    """A project or significant piece of work mentioned in the resume."""

    title: str = Field(..., description="Project name or short description.")
    description: str = Field(
        default="",
        description="What the project did and what the candidate's role was.",
    )
    skills_used: list[str] = Field(
        default_factory=list,
        description="Skills used in this project.",
    )
    is_production: bool = Field(
        default=False,
        description="True if the project was deployed / used in production.",
    )
    has_metrics: bool = Field(
        default=False,
        description="True if the project mentions measurable outcomes.",
    )
    source_snippet: str = Field(
        default="",
        description="The resume sentence(s) this project was extracted from.",
    )


# ---------------------------------------------------------------------------
# Achievement
# ---------------------------------------------------------------------------


class Achievement(BaseModel):
    """A concrete, measurable accomplishment claimed in the resume."""

    description: str = Field(
        ...,
        description="What the candidate claims to have achieved.",
    )
    has_metric: bool = Field(
        default=False,
        description="True if the achievement contains a measurable number or %.",
    )
    metric_snippet: Optional[str] = Field(
        default=None,
        description="The specific metric phrase, e.g. '40% latency reduction'.",
    )
    source_snippet: str = Field(
        default="",
        description="The raw resume sentence this came from.",
    )


# ---------------------------------------------------------------------------
# Career role (timeline entry)
# ---------------------------------------------------------------------------


class CareerRole(BaseModel):
    """One role in the candidate's career timeline."""

    title: str          = Field(..., description="Job title.")
    company: str        = Field(default="", description="Company or organisation name.")
    start_year: Optional[int] = Field(default=None, description="Start year.")
    end_year:   Optional[int] = Field(default=None, description="End year (None = current).")
    duration_years: Optional[float] = Field(
        default=None,
        ge=0,
        description="Duration in years. Computed automatically if start/end provided.",
    )
    category: EmploymentCategory = Field(default=EmploymentCategory.UNKNOWN)
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Key responsibilities in this role, extracted from resume text.",
    )
    is_relevant: bool = Field(
        default=True,
        description="True if this role is relevant to the target JD.",
    )

    @model_validator(mode="after")
    def compute_duration(self) -> "CareerRole":
        if (
            self.duration_years is None
            and self.start_year is not None
            and self.end_year is not None
        ):
            self.duration_years = float(self.end_year - self.start_year)
        return self


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------


class EducationEntry(BaseModel):
    """One education entry from the resume."""

    degree:      str         = Field(default="", description="Degree name.")
    field:       str         = Field(default="", description="Field of study.")
    institution: str         = Field(default="", description="University or institution name.")
    year:        Optional[int] = Field(default=None, description="Graduation year.")
    level:       DegreeLevel = Field(default=DegreeLevel.UNKNOWN)


# ---------------------------------------------------------------------------
# Core output: CandidateProfile
# ---------------------------------------------------------------------------


class CandidateProfile(BaseModel):
    """
    Fully structured profile of one candidate, produced by Layer 4.

    This replaces the raw resume text for all downstream layers.
    Every field is traceable back to specific resume text via source_snippet
    fields on SkillClaim, ProjectEntry, Achievement, etc.

    Consumed by:
      - Layer 5  (Claim Verification)  — verifies skill claims via GitHub
      - Layer 6  (Evidence Ledger)     — stores claims with proof
      - Layer 7  (Skill Graph)         — expands skills with synonyms/graph
      - Layer 8  (Hybrid Retrieval)    — embeds profile for dense search
      - Layer 9  (Multi-Agent Eval)    — evaluates all dimensions
      - Layer 10 (Rubric Scoring)      — scores against HiringProfile
    """

    # Identity
    candidate_id:   str            = Field(..., description="From CandidateInput.")
    name:           str            = Field(default="", description="Candidate full name.")
    email:          Optional[str]  = Field(default=None)
    phone:          Optional[str]  = Field(default=None)

    # Skills — primary signal for matching and verification
    skills: list[SkillClaim] = Field(
        default_factory=list,
        description="All skills extracted, with evidence and confidence.",
    )

    # Work history
    career_timeline: list[CareerRole] = Field(
        default_factory=list,
        description="Chronological list of roles, most recent first.",
    )

    # Projects
    projects: list[ProjectEntry] = Field(
        default_factory=list,
        description="Significant projects mentioned in the resume.",
    )

    # Achievements
    achievements: list[Achievement] = Field(
        default_factory=list,
        description="Concrete, potentially measurable accomplishments.",
    )

    # Education
    education: list[EducationEntry] = Field(
        default_factory=list,
    )

    # Certifications
    certifications: list[str] = Field(
        default_factory=list,
        description="Certification names extracted from the resume.",
    )

    # Computed signals (0.0 – 1.0) used by Layer 10 scoring
    total_years_experience: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total professional experience in years.",
    )
    seniority_signal: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "0–1 signal indicating seniority. Derived from title keywords, "
            "years of experience, and leadership language."
        ),
    )
    leadership_signal: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0–1 signal for leadership/ownership language in the resume.",
    )
    production_signal: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0–1 signal for production/deployment evidence.",
    )
    achievement_signal: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0–1 signal for measurable achievements.",
    )
    career_growth_signal: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0–1 signal for upward career trajectory.",
    )

    # Convenience
    @property
    def skill_names(self) -> list[str]:
        return [s.skill for s in self.skills]

    @property
    def explicit_skill_names(self) -> list[str]:
        return [s.skill for s in self.skills if s.confidence == SkillConfidence.EXPLICIT]

    @property
    def has_production_evidence(self) -> bool:
        return self.production_signal > 0.4

    @property
    def has_leadership_evidence(self) -> bool:
        return self.leadership_signal > 0.4

    @property
    def total_projects(self) -> int:
        return len(self.projects)

    @property
    def total_achievements(self) -> int:
        return len(self.achievements)
