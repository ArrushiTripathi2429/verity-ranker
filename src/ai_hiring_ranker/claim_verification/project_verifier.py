"""
Project-based claim verifier — uses Layer 4 structured project data.

Checks whether skills claimed in the resume are backed by named projects,
production deployments, or measurable outcomes.
"""

from __future__ import annotations

from ..candidate_extraction.schemas import CandidateProfile
from .schemas import EvidenceItem, EvidenceSource


def verify_skills_via_projects(profile: CandidateProfile) -> list[EvidenceItem]:
    """Build evidence items from structured projects and achievements."""
    evidence: list[EvidenceItem] = []

    for project in profile.projects:
        project_text = " ".join(
            [
                project.title,
                project.description,
                project.source_snippet,
                " ".join(project.skills_used),
            ]
        ).lower()

        for skill in profile.skill_names:
            skill_lower = skill.lower()
            if skill_lower not in project_text and not any(
                skill_lower == used.strip().title().lower()
                for used in project.skills_used
            ):
                continue

            relevance = 0.55
            snippet_parts = [f"Project '{project.title}' references {skill}"]
            if project.is_production:
                relevance += 0.15
                snippet_parts.append("production deployment indicated")
            if project.has_metrics:
                relevance += 0.10
                snippet_parts.append("measurable outcomes present")

            evidence.append(
                EvidenceItem(
                    source=EvidenceSource.RESUME,
                    skill=skill,
                    snippet="; ".join(snippet_parts),
                    relevance_score=min(round(relevance, 2), 1.0),
                )
            )

    for achievement in profile.achievements:
        achievement_text = " ".join(
            filter(
                None,
                [
                    achievement.description,
                    achievement.metric_snippet or "",
                    achievement.source_snippet,
                ],
            )
        ).lower()

        for skill in profile.skill_names:
            if skill.lower() not in achievement_text:
                continue
            relevance = 0.65 if achievement.has_metric else 0.50
            evidence.append(
                EvidenceItem(
                    source=EvidenceSource.RESUME,
                    skill=skill,
                    snippet=f"Achievement supports {skill}: {achievement.description[:180]}",
                    relevance_score=relevance,
                )
            )

    return evidence
