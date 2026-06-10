from dataclasses import dataclass

from .candidates import CandidateProfile
from .jd import RoleProfile


@dataclass(frozen=True)
class CandidateScore:
    rank: int
    candidate_id: str
    candidate_name: str
    score: float
    matched_required_skills: list[str]
    matched_preferred_skills: list[str]
    missing_required_skills: list[str]
    evidence_summary: dict[str, list[str]]


def _coverage(candidate_skills: set[str], target_skills: list[str]) -> float:
    if not target_skills:
        return 1.0
    return len(candidate_skills.intersection(target_skills)) / len(target_skills)


def score_candidate(candidate: CandidateProfile, role: RoleProfile, weights: dict[str, float]) -> CandidateScore:
    candidate_skills = set(candidate.skills)
    matched_required = sorted(candidate_skills.intersection(role.required_skills))
    matched_preferred = sorted(candidate_skills.intersection(role.preferred_skills))
    missing_required = sorted(set(role.required_skills) - candidate_skills)

    required_score = _coverage(candidate_skills, role.required_skills)
    preferred_score = _coverage(candidate_skills, role.preferred_skills)

    evidence_count = sum(len(candidate.evidence_snippets.get(skill, [])) for skill in matched_required)
    experience_depth = min(evidence_count / max(len(role.required_skills), 1), 1.0)

    total = (
        weights["required_skill_fit"] * required_score
        + weights["preferred_skill_fit"] * preferred_score
        + weights["experience_depth"] * experience_depth
        + weights["seniority_signal"] * candidate.seniority_signal
        + weights["achievement_signal"] * candidate.achievement_signal
    )

    evidence_summary = {
        skill: candidate.evidence_snippets.get(skill, [])[:2]
        for skill in matched_required + matched_preferred
    }

    return CandidateScore(
        rank=0,
        candidate_id=candidate.candidate_id,
        candidate_name=candidate.name,
        score=round(total * 100, 2),
        matched_required_skills=matched_required,
        matched_preferred_skills=matched_preferred,
        missing_required_skills=missing_required,
        evidence_summary=evidence_summary,
    )


def rank_candidates(candidates: list[CandidateProfile], role: RoleProfile, weights: dict[str, float]) -> list[CandidateScore]:
    scored = [score_candidate(candidate, role, weights) for candidate in candidates]
    scored.sort(key=lambda item: (-item.score, item.candidate_id))
    return [
        CandidateScore(
            rank=index,
            candidate_id=item.candidate_id,
            candidate_name=item.candidate_name,
            score=item.score,
            matched_required_skills=item.matched_required_skills,
            matched_preferred_skills=item.matched_preferred_skills,
            missing_required_skills=item.missing_required_skills,
            evidence_summary=item.evidence_summary,
        )
        for index, item in enumerate(scored, start=1)
    ]

