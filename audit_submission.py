#!/usr/bin/env python3
"""
Run Layer 13 (Fairness + Proxy Audit) and Layer 14 (Rank Stability Test)
on a completed precompute cache. Dev-time only — not part of rank.py.

Usage:
  python audit_submission.py --candidates candidates.jsonl --cache-dir cache --output audit_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.hackathon.fairness_stability import (  # noqa: E402
    run_fairness_audit,
    run_stability_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", default=Path("audit_report.json"), type=Path)
    parser.add_argument("--n-runs", type=int, default=5)
    args = parser.parse_args()

    cache_path = args.cache_dir / "candidate_features.jsonl"

    print("Running fairness audit (full population)...")
    fairness = run_fairness_audit(args.candidates, cache_path)

    print(f"Running stability audit ({args.n_runs} perturbation runs)...")
    stability = run_stability_audit(cache_path, n_runs=args.n_runs)

    report = {"fairness": fairness, "stability": stability}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nFairness risk level: {fairness['overall_risk_level']}")
    print(f"Flags: {len(fairness['flags'])}")
    print(f"Stability: {stability['stable_count']}/{stability['evaluated_top_k']} stable")
    print(f"Unstable candidate_ids: {stability['unstable_ids']}")
    print(f"\nFull report → {args.output}")


if __name__ == "__main__":
    main()