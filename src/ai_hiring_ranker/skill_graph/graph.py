"""
Skill & Role Knowledge Graph engine — Layer 7.

Builds an in-memory graph from graph_data.py and exposes a single public
function:

    expand_skills(skill_names, max_hops=1, min_weight=0.40)
        → GraphExpansionResult

The graph is loaded once (module-level singleton via @lru_cache) so every
pipeline run uses the same object without re-building it.

Algorithm
---------
1. Resolve each raw skill name to its canonical node via the alias index.
2. Breadth-first traversal from each seed node up to max_hops hops.
3. Each edge's weight is decayed by hop distance so distant expansions
   contribute less to downstream scoring.
4. Results are deduplicated by canonical name — the highest-weight path wins.
5. Seeds that don't resolve to any node are returned as ExpandedSkills with
   weight=1.0 and hop_distance=0 so they're still usable in retrieval.

Public API
----------
from ai_hiring_ranker.skill_graph import expand_skills, get_graph
"""

from __future__ import annotations

import logging
from collections import deque
from functools import lru_cache
from typing import Optional

from .graph_data import _RAW_NODES
from .schemas import (
    ExpandedSkill,
    GraphExpansionResult,
    RelationshipType,
    SkillEdge,
    SkillNode,
)

logger = logging.getLogger(__name__)

# Weight decay applied per hop during BFS expansion.
# hop 0 (seed)   → weight × 1.00
# hop 1          → weight × _HOP_DECAY
# hop 2          → weight × _HOP_DECAY²
_HOP_DECAY = 0.70


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------


class SkillGraph:
    """
    In-memory undirected skill graph with O(1) lookup by canonical name
    and O(1) alias resolution.

    Built once from _RAW_NODES via _build_graph(); accessed via get_graph().
    """

    def __init__(self, nodes: list[SkillNode]) -> None:
        # canonical_name (title-cased) → SkillNode
        self._nodes: dict[str, SkillNode] = {}
        # alias (lowercased) → canonical_name
        self._alias_index: dict[str, str] = {}

        for node in nodes:
            self._nodes[node.canonical_name] = node
            # Register the canonical name itself as an alias (lowercased)
            self._alias_index[node.canonical_name.lower()] = node.canonical_name
            for alias in node.aliases:
                self._alias_index[alias.lower()] = node.canonical_name

        # Add reverse edges for bidirectional relationships
        self._add_reverse_edges()

        logger.debug(
            "SkillGraph built: %d nodes, %d aliases",
            len(self._nodes),
            len(self._alias_index),
        )

    # ── internal ────────────────────────────────────────────────────────

    def _add_reverse_edges(self) -> None:
        """
        For every edge marked bidirectional=True, add the reverse edge
        on the target node if it doesn't already exist.
        """
        additions: list[tuple[str, SkillEdge]] = []

        for canonical, node in self._nodes.items():
            for edge in node.edges:
                if not edge.bidirectional:
                    continue
                target_node = self._nodes.get(edge.target)
                if target_node is None:
                    continue
                # Check if the reverse edge already exists
                already = any(
                    e.target == canonical and e.relationship == edge.relationship
                    for e in target_node.edges
                )
                if not already:
                    reverse = SkillEdge(
                        target=canonical,
                        relationship=edge.relationship,
                        weight=edge.weight,
                        bidirectional=False,  # Avoid infinite recursion
                    )
                    additions.append((edge.target, reverse))

        for canonical_target, reverse_edge in additions:
            node = self._nodes[canonical_target]
            node.edges.append(reverse_edge)

    # ── public lookup ────────────────────────────────────────────────────

    def resolve(self, raw_name: str) -> Optional[str]:
        """
        Resolve a raw/alias skill name to its canonical name.
        Returns None if no node is found.
        """
        return self._alias_index.get(raw_name.strip().lower())

    def get_node(self, canonical: str) -> Optional[SkillNode]:
        return self._nodes.get(canonical)

    @property
    def all_canonical_names(self) -> list[str]:
        return list(self._nodes.keys())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # ── BFS expansion ────────────────────────────────────────────────────

    def expand(
        self,
        raw_skills: list[str],
        *,
        max_hops: int = 1,
        min_weight: float = 0.40,
        relationship_filter: Optional[set[RelationshipType]] = None,
    ) -> GraphExpansionResult:
        """
        Expand a list of raw skill names through the graph via BFS.

        Args:
            raw_skills:           Input skill names (aliases accepted).
            max_hops:             Maximum BFS depth from each seed.
                                  1 = direct neighbours only (default).
                                  2 = neighbours of neighbours.
            min_weight:           Discard expansions below this weight threshold.
            relationship_filter:  If given, only traverse edges of these types.
                                  None = traverse all relationship types.

        Returns:
            GraphExpansionResult with expanded skills sorted by weight desc.
        """
        if relationship_filter is None:
            relationship_filter = {
                RelationshipType.SYNONYM,
                RelationshipType.ADJACENT,
                RelationshipType.TRANSFERABLE,
                RelationshipType.SUBSET,
                RelationshipType.SUPERSET,
            }

        # best_weight[canonical] = highest weight path seen so far
        best_weight: dict[str, float] = {}
        # best_entry[canonical] = ExpandedSkill with the best path
        best_entry: dict[str, ExpandedSkill] = {}

        unresolved: list[str] = []
        seed_skills = [s.strip() for s in raw_skills if s.strip()]

        # ── BFS from each seed ────────────────────────────────────────
        for raw in seed_skills:
            canonical = self.resolve(raw)

            if canonical is None:
                # Not in graph — keep as-is with full weight
                unresolved.append(raw.title())
                key = raw.title()
                if best_weight.get(key, -1) < 1.0:
                    best_weight[key] = 1.0
                    best_entry[key] = ExpandedSkill(
                        raw_name=raw,
                        canonical=key,
                        relationship=RelationshipType.SYNONYM,
                        weight=1.0,
                        hop_distance=0,
                    )
                continue

            # Seed itself (hop 0, weight 1.0)
            if best_weight.get(canonical, -1) < 1.0:
                best_weight[canonical] = 1.0
                best_entry[canonical] = ExpandedSkill(
                    raw_name=raw,
                    canonical=canonical,
                    relationship=RelationshipType.SYNONYM,
                    weight=1.0,
                    hop_distance=0,
                )

            if max_hops == 0:
                continue

            # BFS queue: (current_canonical, hop_distance, accumulated_weight, rel_type)
            queue: deque[tuple[str, int, float, RelationshipType]] = deque()
            queue.append((canonical, 0, 1.0, RelationshipType.SYNONYM))
            visited_from_seed: set[str] = {canonical}

            while queue:
                current, hop, current_weight, _ = queue.popleft()
                if hop >= max_hops:
                    continue

                node = self._nodes.get(current)
                if node is None:
                    continue

                for edge in node.edges:
                    if edge.relationship not in relationship_filter:
                        continue

                    next_canonical = edge.target
                    if next_canonical not in self._nodes:
                        continue
                    if next_canonical in visited_from_seed:
                        continue

                    # Decay weight by hop and edge weight
                    new_weight = round(
                        current_weight * edge.weight * (_HOP_DECAY ** hop),
                        4,
                    )
                    if new_weight < min_weight:
                        continue

                    visited_from_seed.add(next_canonical)

                    # Keep best (highest-weight) path to this canonical skill
                    if new_weight > best_weight.get(next_canonical, -1):
                        best_weight[next_canonical] = new_weight
                        best_entry[next_canonical] = ExpandedSkill(
                            raw_name=raw,
                            canonical=next_canonical,
                            relationship=edge.relationship,
                            weight=new_weight,
                            hop_distance=hop + 1,
                        )

                    queue.append((next_canonical, hop + 1, new_weight, edge.relationship))

        # ── Assemble result ───────────────────────────────────────────
        expanded = sorted(best_entry.values(), key=lambda x: x.weight, reverse=True)

        return GraphExpansionResult(
            seed_skills=seed_skills,
            expanded_skills=expanded,
            unresolved=[u for u in unresolved if u not in {e.canonical for e in expanded}],
        )


# ---------------------------------------------------------------------------
# Module-level singleton (built once, reused)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_graph() -> SkillGraph:
    """Return the singleton SkillGraph, building it on first call."""
    graph = SkillGraph(_RAW_NODES)
    logger.info("SkillGraph loaded: %d canonical skills", graph.node_count)
    return graph


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


def expand_skills(
    skill_names: list[str],
    *,
    max_hops: int = 1,
    min_weight: float = 0.40,
    relationship_filter: Optional[set[RelationshipType]] = None,
) -> GraphExpansionResult:
    """
    Expand a list of raw skill names through the knowledge graph.

    This is the main entry point for Layer 8 (Hybrid Retrieval) and
    Layer 9 (Multi-Agent Evaluation).

    Args:
        skill_names:          Raw/alias skill names from a JD or candidate profile.
        max_hops:             BFS depth. 1 = direct neighbours (recommended default).
                              Use 2 for broad expansion (more recall, less precision).
        min_weight:           Minimum edge weight to include in results (0.0–1.0).
                              0.40 keeps moderately-adjacent skills; raise to 0.60
                              for tighter expansion.
        relationship_filter:  Limit expansion to specific relationship types.
                              None = all types (synonym + adjacent + transferable).

    Returns:
        GraphExpansionResult — sorted by weight descending.

    Example:
        >>> result = expand_skills(["python", "fastapi"])
        >>> result.all_canonical_names
        ['Python', 'FastAPI', 'Pydantic', 'Flask', 'REST API', ...]
        >>> result.get_weight("Pydantic")
        0.68
    """
    return get_graph().expand(
        skill_names,
        max_hops=max_hops,
        min_weight=min_weight,
        relationship_filter=relationship_filter,
    )


def resolve_skill(raw_name: str) -> Optional[str]:
    """
    Resolve a single raw/alias name to its canonical form.
    Returns None if not in the graph.

    Example:
        >>> resolve_skill("sklearn")
        'Scikit-Learn'
        >>> resolve_skill("pyspark")
        'Apache Spark'
    """
    return get_graph().resolve(raw_name)
