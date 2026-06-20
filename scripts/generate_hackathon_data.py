#!/usr/bin/env python3
"""Generate synthetic hackathon candidates.jsonl for local testing."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

GOOD_SKILLS = [
    "Python", "machine learning", "FastAPI", "Docker", "model evaluation", "testing",
    "embeddings", "SQL", "pandas", "scikit-learn",
]
BUZZWORDS = [
    "LLM", "RAG", "transformers", "GenAI", "LangChain", "vector database",
    "Kubernetes", "microservices", "blockchain", "Web3", "quantum computing",
]


def _date(days_ago: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.date().isoformat()


def generate(path: Path, count: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        # Strong candidates
        for i in range(1, count // 2):
            cid = f"C{i:05d}"
            skills = rng.sample(GOOD_SKILLS, k=rng.randint(5, 8))
            record = {
                "candidate_id": cid,
                "job_title": rng.choice(
                    ["Machine Learning Engineer", "ML Engineer", "Backend ML Engineer"]
                ),
                "years_experience": rng.randint(3, 10),
                "skills": skills,
                "github_activity_score": round(rng.uniform(0.35, 0.95), 2),
                "last_active_date": _date(rng.randint(1, 60)),
                "recruiter_response_rate": round(rng.uniform(0.5, 0.95), 2),
                "resume_text": (
                    f"Candidate ID: {cid}\n"
                    f"Machine Learning Engineer with {rng.randint(3, 8)} years building "
                    f"Python ML services. Deployed FastAPI inference APIs with Docker. "
                    f"Built model evaluation pipelines and tests. "
                    f"Skills: {', '.join(skills)}."
                ),
            }
            fh.write(json.dumps(record) + "\n")

        # Weak / low engagement
        for i in range(count // 2, count - 20):
            cid = f"C{i:05d}"
            skills = rng.sample(GOOD_SKILLS, k=rng.randint(2, 4))
            record = {
                "candidate_id": cid,
                "job_title": "Junior Data Analyst",
                "years_experience": rng.randint(0, 2),
                "skills": skills,
                "github_activity_score": round(rng.uniform(0.0, 0.25), 2),
                "last_active_date": _date(rng.randint(400, 900)),
                "recruiter_response_rate": round(rng.uniform(0.0, 0.15), 2),
                "resume_text": (
                    f"Candidate ID: {cid}\n"
                    f"Junior candidate with internship experience in notebooks. "
                    f"Skills: {', '.join(skills)}."
                ),
            }
            fh.write(json.dumps(record) + "\n")

        # Honeypots
        honeypots = [
            {
                "candidate_id": "HONEYPOT01",
                "job_title": "Intern",
                "years_experience": 25,
                "skills": GOOD_SKILLS + BUZZWORDS,
                "github_activity_score": 0.0,
                "last_active_date": _date(10),
                "recruiter_response_rate": 0.9,
                "resume_text": "I invented Python and have 50+ years experience as an intern.",
            },
            {
                "candidate_id": "HONEYPOT02",
                "job_title": "Student",
                "years_experience": 40,
                "skills": BUZZWORDS * 3,
                "github_activity_score": 0.01,
                "last_active_date": _date(5),
                "recruiter_response_rate": 0.8,
                "resume_text": " ".join(BUZZWORDS) + " Nobel prize CEO of Google.",
            },
        ]
        for hp in honeypots:
            fh.write(json.dumps(hp) + "\n")

        # Keyword stuffers
        for j in range(18):
            cid = f"STUFF{j:03d}"
            record = {
                "candidate_id": cid,
                "job_title": "Marketing Coordinator",
                "years_experience": 1,
                "skills": BUZZWORDS,
                "github_activity_score": 0.05,
                "last_active_date": _date(20),
                "recruiter_response_rate": 0.7,
                "resume_text": " ".join(BUZZWORDS) + " expert in everything AI.",
            }
            fh.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/hackathon/candidates.jsonl"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.count, args.seed)
    print(f"Wrote {args.count + 20} rows to {args.output}")


if __name__ == "__main__":
    main()
