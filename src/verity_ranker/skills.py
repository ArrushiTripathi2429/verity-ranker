SKILL_ALIASES = {
    "python": ["python"],
    "machine learning": ["machine learning", "ml", "supervised learning"],
    "model evaluation": ["model evaluation", "evaluated classification", "metrics", "evaluation pipelines"],
    "fastapi": ["fastapi"],
    "production engineering": ["production engineering", "production-ready", "reliable", "testable systems"],
    "docker": ["docker", "docker-based", "docker images"],
    "retrieval": ["retrieval", "vector search", "search"],
    "embeddings": ["embeddings", "embedding"],
    "cloud deployment": ["cloud deployment", "deployments", "deployment"],
    "testing": ["tests", "testing", "testable"],
    "ci/cd": ["ci/cd", "ci pipelines"],
    "sql": ["sql"],
    "dashboards": ["dashboard", "dashboards"],
}


def extract_skills(text: str) -> set[str]:
    lowered = text.lower()
    found = set()
    for skill, aliases in SKILL_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            found.add(skill)
    return found

