"""
Layer 7 — Skill & Role Knowledge Graph.

Expands raw skill names from JDs and candidate profiles into a richer
network of synonyms, adjacent skills, transferable skills, and role
hierarchies. Used by Layer 8 (Hybrid Retrieval) to enrich queries and
by Layer 9 (Multi-Agent Eval) for partial-credit skill matching.

Public API
----------
from ai_hiring_ranker.skill_graph import (
    # Main entry point
    expand_skills,       # list[str] → GraphExpansionResult
    resolve_skill,       # str → Optional[str] (canonical name)
    get_graph,           # () → SkillGraph singleton

    # Schema types
    GraphExpansionResult,
    ExpandedSkill,
    SkillNode,
    SkillEdge,
    RelationshipType,
)
"""

from .graph import expand_skills, get_graph, resolve_skill
from .schemas import (
    ExpandedSkill,
    GraphExpansionResult,
    RelationshipType,
    SkillEdge,
    SkillNode,
)

__all__ = [
    # Functions
    "expand_skills",
    "resolve_skill",
    "get_graph",
    # Schema types
    "GraphExpansionResult",
    "ExpandedSkill",
    "SkillNode",
    "SkillEdge",
    "RelationshipType",
]
