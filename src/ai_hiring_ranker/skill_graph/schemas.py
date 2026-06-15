"""
Output schemas for Layer 7 — Skill & Role Knowledge Graph.

The graph expands raw skill names into a richer network of:
  - canonical names + synonyms (so "ML" and "machine learning" are one node)
  - adjacent / related skills (Python → FastAPI, Pandas, NumPy)
  - transferable skills (Spark → Hadoop, Flink share a family)
  - role hierarchy (Junior ML Engineer → ML Engineer → Senior → Staff)

Consumed by:
  - Layer 8  (Hybrid Retrieval)  — graph-expanded query enrichment
  - Layer 9  (Multi-Agent Eval)  — Technical Fit Agent uses adjacency
  - Layer 10 (Rubric Scoring)    — partial credit for adjacent skills

Design decisions:
  - SkillNode is the fundamental unit; the graph is a flat dict keyed by
    canonical name for O(1) lookup instead of a heavy graph library.
  - GraphExpansionResult is what callers actually use — given a list of
    raw skill names it returns the canonical set plus all expanded skills
    with their relationship type and confidence weight.
  - Everything is pure Python + Pydantic, no networkx or neo4j needed.
    Layer 7 is intentionally self-contained and importable offline.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Relationship type between two skills
# ---------------------------------------------------------------------------


class RelationshipType(str, Enum):
    """How two skills are related in the graph."""

    SYNONYM        = "synonym"       # different name, same skill (ML ↔ machine learning)
    ADJACENT       = "adjacent"      # commonly co-used (Python → FastAPI)
    TRANSFERABLE   = "transferable"  # experience in A strongly implies capacity in B
    SUBSET         = "subset"        # A is a subset/specialisation of B (Keras ⊂ Deep Learning)
    SUPERSET       = "superset"      # A is a generalisation of B (Deep Learning ⊃ Keras)
    ROLE_HIERARCHY = "role_hierarchy" # seniority ladder (ML Eng → Senior ML Eng)


# ---------------------------------------------------------------------------
# Skill edge — a directed relationship to another skill
# ---------------------------------------------------------------------------


class SkillEdge(BaseModel):
    """A directed edge from one skill to a related skill."""

    target:           str              = Field(..., description="Canonical name of the related skill.")
    relationship:     RelationshipType = Field(..., description="How the two skills are related.")
    weight:           float            = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Strength of the relationship (0–1). "
            "1.0 = perfect synonym, 0.3 = loosely related."
        ),
    )
    bidirectional:    bool             = Field(
        default=False,
        description="True if the relationship holds in both directions.",
    )

    @field_validator("target", mode="before")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().title()


# ---------------------------------------------------------------------------
# Skill node — one canonical skill with all its metadata
# ---------------------------------------------------------------------------


class SkillNode(BaseModel):
    """
    One node in the skill graph.

    The canonical_name is the normalised, title-cased name used as the
    graph key. All aliases resolve to this canonical name before lookup.
    """

    canonical_name: str            = Field(..., description="The normalised, canonical skill name.")
    aliases:        list[str]      = Field(
        default_factory=list,
        description=(
            "Alternative names / abbreviations that map to this skill. "
            "e.g. ['ML', 'machine learning', 'supervised learning'] → 'Machine Learning'."
        ),
    )
    category:       str            = Field(
        default="general",
        description=(
            "Skill category for grouping: programming_language, framework, "
            "platform, ml_concept, data_engineering, devops, soft_skill, domain."
        ),
    )
    esco_uri:       Optional[str]  = Field(
        default=None,
        description="ESCO skill URI for interoperability (optional).",
    )
    onet_id:        Optional[str]  = Field(
        default=None,
        description="O*NET skill ID for interoperability (optional).",
    )
    edges:          list[SkillEdge] = Field(
        default_factory=list,
        description="All outgoing edges from this skill node.",
    )

    @field_validator("canonical_name", mode="before")
    @classmethod
    def normalise_name(cls, v: str) -> str:
        return v.strip().title()

    @field_validator("aliases", mode="before")
    @classmethod
    def normalise_aliases(cls, v: list[str]) -> list[str]:
        return [a.strip().lower() for a in v]

    @property
    def adjacent_skills(self) -> list[str]:
        return [
            e.target for e in self.edges
            if e.relationship == RelationshipType.ADJACENT
        ]

    @property
    def synonyms(self) -> list[str]:
        return [
            e.target for e in self.edges
            if e.relationship == RelationshipType.SYNONYM
        ]

    @property
    def transferable_skills(self) -> list[str]:
        return [
            e.target for e in self.edges
            if e.relationship == RelationshipType.TRANSFERABLE
        ]

    def get_edges_by_type(self, rel: RelationshipType) -> list[SkillEdge]:
        return [e for e in self.edges if e.relationship == rel]


# ---------------------------------------------------------------------------
# Expanded skill — one skill after graph traversal
# ---------------------------------------------------------------------------


class ExpandedSkill(BaseModel):
    """
    A single skill produced by graph expansion.

    Carries the original (raw) name, the resolved canonical name,
    how it was reached, and a confidence weight for downstream scoring.
    """

    raw_name:      str              = Field(..., description="The skill name as it appeared in the input.")
    canonical:     str              = Field(..., description="The graph-resolved canonical name.")
    relationship:  RelationshipType = Field(
        default=RelationshipType.SYNONYM,
        description="How this skill was reached from the seed.",
    )
    weight:        float            = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Matching weight for scoring (1.0 = direct / exact, "
            "lower = expanded via adjacency or transferability)."
        ),
    )
    hop_distance:  int              = Field(
        default=0,
        ge=0,
        description="Graph hops from the seed skill (0 = seed itself).",
    )

    @field_validator("canonical", "raw_name", mode="before")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().title()

    @property
    def is_exact(self) -> bool:
        return self.hop_distance == 0

    @property
    def is_adjacent(self) -> bool:
        return self.relationship == RelationshipType.ADJACENT and self.hop_distance == 1

    @property
    def is_transferable(self) -> bool:
        return self.relationship == RelationshipType.TRANSFERABLE


# ---------------------------------------------------------------------------
# Graph expansion result — output of expand_skills()
# ---------------------------------------------------------------------------


class GraphExpansionResult(BaseModel):
    """
    Result of expanding a set of raw skill names through the graph.

    This is what Layer 8 (Hybrid Retrieval) and Layer 9 (Multi-Agent Eval)
    actually consume. The expanded_skills list is a superset of the seeds,
    ordered by weight descending so high-confidence matches come first.
    """

    seed_skills:      list[str]           = Field(
        default_factory=list,
        description="The raw input skill names before expansion.",
    )
    expanded_skills:  list[ExpandedSkill] = Field(
        default_factory=list,
        description="All skills after graph traversal, seeds + expansions.",
    )
    unresolved:       list[str]           = Field(
        default_factory=list,
        description="Seed skills that had no node in the graph (returned as-is with weight=1.0).",
    )

    @property
    def all_canonical_names(self) -> list[str]:
        """Deduplicated list of canonical names, highest-weight first."""
        seen: set[str] = set()
        out: list[str] = []
        for s in sorted(self.expanded_skills, key=lambda x: x.weight, reverse=True):
            if s.canonical not in seen:
                seen.add(s.canonical)
                out.append(s.canonical)
        return out

    @property
    def exact_matches(self) -> list[ExpandedSkill]:
        return [s for s in self.expanded_skills if s.is_exact]

    @property
    def adjacent_matches(self) -> list[ExpandedSkill]:
        return [s for s in self.expanded_skills if s.is_adjacent]

    @property
    def transferable_matches(self) -> list[ExpandedSkill]:
        return [s for s in self.expanded_skills if s.is_transferable]

    def get_weight(self, canonical: str) -> float:
        """Return the highest weight assigned to a canonical skill name."""
        matches = [s.weight for s in self.expanded_skills if s.canonical == canonical]
        return max(matches, default=0.0)

    def summary(self) -> str:
        return (
            f"seeds={len(self.seed_skills)}  "
            f"expanded={len(self.expanded_skills)}  "
            f"unresolved={len(self.unresolved)}"
        )
