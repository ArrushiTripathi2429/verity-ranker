"""
Graph-expansion skill retrieval for Layer 8.

Uses the Layer 7 SkillGraph to expand the JD's required/preferred skills
into a richer set (synonyms, adjacent, transferable) then scores how well
each candidate's skill set covers the expanded query.

This catches matches that BM25 and dense retrieval both miss:
  - "ML" ↔ "Machine Learning" (synonym)
  - JD asks for "FastAPI" → candidate has "Flask" (adjacent, weight 0.70)
  - JD asks for "Apache Spark" → candidate has "Hadoop" (transferable, 0.55)

Partial credit:
  Each expanded skill carries a weight (0–1) that reflects how strong
  the graph relationship is. Candidates get partial credit for adjacent/
  transferable skills, not full credit, so the score is always justified.

Public API
----------
score_graph(candidates, hiring_profile) → dict[candidate_id, float]
match_skills_graph(candidate_skills, jd_expanded) → tuple[list, list, float]
"""

from __future__ import annotations

import logging
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile
from ..jd_intelligence.schemas import HiringProfile
from ..skill_graph.graph import expand_skills
from ..skill_graph.schemas import ExpandedSkill, GraphExpansionResult

logger = logging.getLogger(__name__)

# Weight applied to required skill matches vs preferred skill matches
_REQUIRED_WEIGHT  = 1.0
_PREFERRED_WEIGHT = 0.50

# BFS expansion settings
_MAX_HOPS   = 1      # 1 hop = direct neighbours only (keeps precision high)
_MIN_WEIGHT = 0.40   # don't include very loosely related skills


# ---------------------------------------------------------------------------
# Core skill matching logic
# ---------------------------------------------------------------------------


def match_skills_graph(
    candidate_skill_names: list[str],
    jd_expanded: GraphExpansionResult,
) -> tuple[list[str], list[str], float]:
    """
    Match a candidate's skills against the graph-expanded JD skill set.

    Args:
        candidate_skill_names:  Canonical skill names from CandidateProfile.
        jd_expanded:            GraphExpansionResult from expanding JD skills.

    Returns:
        Tuple of:
          - exact_matches:    Skills matched exactly (or via synonym)
          - expanded_matches: Skills matched via graph expansion
          - weighted_score:   Sum of match weights / max possible score (0–1)
    """
    if not candidate_skill_names or not jd_expanded.expanded_skills:
        return [], [], 0.0

    # Build a lookup: canonical_name → best weight in the expanded JD
    jd_skill_weights: dict[str, float] = {}
    for exp in jd_expanded.expanded_skills:
        current = jd_skill_weights.get(exp.canonical, 0.0)
        jd_skill_weights[exp.canonical] = max(current, exp.weight)

    # Normalise candidate skills for lookup
    cand_set = {s.strip().title() for s in candidate_skill_names}

    # Also expand the candidate's skills one hop to catch synonym matches
    cand_expanded = expand_skills(
        candidate_skill_names,
        max_hops=1,
        min_weight=0.70,  # only very close synonyms when expanding candidate side
    )
    cand_canonical_set = {e.canonical for e in cand_expanded.expanded_skills}

    exact_matches: list[str] = []
    expanded_matches: list[str] = []
    total_score = 0.0

    for jd_canonical, jd_weight in jd_skill_weights.items():
        if jd_canonical in cand_set or jd_canonical in cand_canonical_set:
            # Check if it's an exact match (hop_distance=0) or expanded
            is_exact = jd_canonical in cand_set
            if is_exact:
                exact_matches.append(jd_canonical)
            else:
                expanded_matches.append(jd_canonical)
            total_score += jd_weight

    # Max possible score = sum of all JD expanded weights (perfect coverage)
    max_possible = sum(jd_skill_weights.values()) or 1.0
    normalised = round(total_score / max_possible, 4)

    return exact_matches, expanded_matches, min(1.0, normalised)


# ---------------------------------------------------------------------------
# Per-candidate scoring
# ---------------------------------------------------------------------------


def _expand_jd_skills(hiring_profile: HiringProfile) -> tuple[GraphExpansionResult, GraphExpansionResult]:
    """
    Expand required and preferred JD skills separately through the graph.
    Returns (required_expanded, preferred_expanded).
    """
    required_expanded  = expand_skills(
        hiring_profile.all_required_skill_names,
        max_hops=_MAX_HOPS,
        min_weight=_MIN_WEIGHT,
    )
    preferred_expanded = expand_skills(
        hiring_profile.all_preferred_skill_names,
        max_hops=_MAX_HOPS,
        min_weight=_MIN_WEIGHT,
    )
    return required_expanded, preferred_expanded


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def score_graph(
    candidates: list[CandidateProfile],
    hiring_profile: HiringProfile,
) -> dict[str, float]:
    """
    Score every candidate using graph-expanded skill matching.

    The final score blends required and preferred skill coverage,
    with required skills weighted more heavily.

    Args:
        candidates:      CandidateProfiles from Layer 4.
        hiring_profile:  HiringProfile from Layer 2.

    Returns:
        dict mapping candidate_id → graph match score (0–1).
    """
    if not candidates:
        return {}

    required_expanded, preferred_expanded = _expand_jd_skills(hiring_profile)

    logger.debug(
        "Graph expansion: %d required seeds → %d expanded  |  "
        "%d preferred seeds → %d expanded",
        len(hiring_profile.all_required_skill_names),
        len(required_expanded.expanded_skills),
        len(hiring_profile.all_preferred_skill_names),
        len(preferred_expanded.expanded_skills),
    )

    scores: dict[str, float] = {}

    for candidate in candidates:
        skill_names = candidate.skill_names

        _, _, req_score  = match_skills_graph(skill_names, required_expanded)
        _, _, pref_score = match_skills_graph(skill_names, preferred_expanded)

        # Blend: required skills dominate
        combined = round(
            _REQUIRED_WEIGHT  * req_score +
            _PREFERRED_WEIGHT * pref_score,
            4,
        )
        # Normalise by the total possible (1.0 + 0.5 = 1.5)
        normalised = round(combined / (_REQUIRED_WEIGHT + _PREFERRED_WEIGHT), 4)
        scores[candidate.candidate_id] = min(1.0, normalised)

    return scores


def get_matched_skills(
    candidate: CandidateProfile,
    hiring_profile: HiringProfile,
) -> tuple[list[str], list[str], list[str]]:
    """
    Return the specific matched skills for one candidate (for audit trail).

    Returns:
        Tuple of:
          - matched_required:   Required JD skills found in candidate
          - matched_preferred:  Preferred JD skills found in candidate
          - graph_expanded:     Skills matched only via graph expansion
    """
    required_expanded, preferred_expanded = _expand_jd_skills(hiring_profile)
    skill_names = candidate.skill_names

    req_exact, req_expanded, _  = match_skills_graph(skill_names, required_expanded)
    pref_exact, pref_expanded, _ = match_skills_graph(skill_names, preferred_expanded)

    matched_required  = list(dict.fromkeys(req_exact))
    matched_preferred = list(dict.fromkeys(pref_exact))
    graph_expanded    = list(dict.fromkeys(req_expanded + pref_expanded))

    return matched_required, matched_preferred, graph_expanded
