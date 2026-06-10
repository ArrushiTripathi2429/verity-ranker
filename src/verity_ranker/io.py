import csv
import json
from dataclasses import asdict
from pathlib import Path

from .jd import RoleProfile
from .scoring import CandidateScore


CSV_COLUMNS = [
    "rank",
    "candidate_id",
    "candidate_name",
    "score",
    "matched_required_skills",
    "matched_preferred_skills",
    "missing_required_skills",
]


def load_weights(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_ranked_output(path: Path, scores: list[CandidateScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "rank": score.rank,
                    "candidate_id": score.candidate_id,
                    "candidate_name": score.candidate_name,
                    "score": f"{score.score:.2f}",
                    "matched_required_skills": "; ".join(score.matched_required_skills),
                    "matched_preferred_skills": "; ".join(score.matched_preferred_skills),
                    "missing_required_skills": "; ".join(score.missing_required_skills),
                }
            )


def write_audit_report(path: Path, role: RoleProfile, scores: list[CandidateScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v1_basic_ranker",
        "limitations": [
            "No GitHub proof-of-work verification.",
            "No evidence ledger with stable evidence IDs.",
            "No fairness or rank-stability audit.",
            "No listwise reranking.",
        ],
        "role_profile": asdict(role),
        "ranked_candidates": [asdict(score) for score in scores],
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

