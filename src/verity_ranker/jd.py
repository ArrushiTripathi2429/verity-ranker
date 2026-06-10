from dataclasses import dataclass

from .skills import extract_skills
from .text import split_sentences


@dataclass(frozen=True)
class RoleProfile:
    required_skills: list[str]
    preferred_skills: list[str]
    responsibilities: list[str]
    ambiguity_flags: list[str]


def parse_jd(text: str) -> RoleProfile:
    required: set[str] = set()
    preferred: set[str] = set()
    responsibilities: list[str] = []
    ambiguity_flags: list[str] = []

    for sentence in split_sentences(text):
        skills = extract_skills(sentence)
        lowered = sentence.lower()

        if "preferred" in lowered or "nice to have" in lowered:
            preferred.update(skills)
        else:
            required.update(skills)

        if any(term in lowered for term in ["build", "deploy", "evaluate", "reliable", "testable"]):
            responsibilities.append(sentence)

        if any(term in lowered for term in ["etc", "and more", "strong background"]):
            ambiguity_flags.append(sentence)

    preferred -= required

    return RoleProfile(
        required_skills=sorted(required),
        preferred_skills=sorted(preferred),
        responsibilities=responsibilities,
        ambiguity_flags=ambiguity_flags,
    )

