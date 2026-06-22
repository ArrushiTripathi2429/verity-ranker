#!/usr/bin/env python3
"""
Layer 17 — Evaluation + Ablation Report.

Reconstructs 5 progressively richer rankings from the SAME precompute
cache (no new LLM calls, no new precompute run) and measures how each
added layer changes the top-100's composition.

Usage:
  python ablation_report.py --cache-dir cache --output docs/ablation_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = {
    "skill_fit": 0.30, "experience_depth": 0.20,
    "seniority_match": 0.15, "domain_match": 0.15,
    "career_growth": 0.10, "proof_strength": 0.10,
}

OFF_DOMAIN_TITLE_RE = re.compile(r"\b(Backend|Analytics|Data)\s+Engineer\b", re.I)


def load_jd_skill_count(jd_cache_path: Path) -> int:
    jd = json.loads(jd_cache_path.read_text(encoding="utf-8"))
    req = jd.get("all_required_skill_names", []) or []
    pref = jd.get("all_preferred_skill_names", []) or []
    return max(1, len(req) + len(pref))


def load_cache(cache_path: Path) -> list[dict[str, Any]]:
    rows = []
    with cache_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tier_scores(row: dict[str, Any], jd_skill_count: int) -> dict[str, float]:
    dims = row.get("dimensions") or {}

    t1 = float(row.get("keyword_score", 0.0))

    n_req = len(row.get("matched_required", []) or [])
    n_pref = len(row.get("matched_preferred", []) or [])
    n_partial = len(row.get("partial_matches", []) or [])
    t2 = min(1.0, (n_req + n_pref + 0.5 * n_partial) / jd_skill_count)

    t3 = sum(float(dims.get(k, 0.0)) * w for k, w in DEFAULT_WEIGHTS.items())

    t4 = 0.85 * t3 + 0.15 * t1

    t5 = float(row.get("final_score", 0.0)) / 100.0

    return {
        "keyword_only": round(t1, 4),
        "skill_graph": round(t2, 4),
        "agent_rubric": round(t3, 4),
        "retrieval_blend": round(t4, 4),
        "full_pipeline": round(t5, 4),
    }


def top_k_ids(rows, tier, scores, k=100):
    ranked = sorted(rows, key=lambda r: (-scores[r["candidate_id"]][tier], r["candidate_id"]))
    return [r["candidate_id"] for r in ranked[:k]]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    return round(len(sa & sb) / len(union), 3) if union else 0.0


def off_domain_count(rows_by_id, ids):
    return sum(1 for cid in ids if OFF_DOMAIN_TITLE_RE.search(rows_by_id[cid].get("job_title") or ""))


def transition_count(rows_by_id, ids):
    return sum(1 for cid in ids if rows_by_id[cid].get("transition_multiplier", 1.0) < 1.0)


def avg_risk(rows_by_id, ids, field):
    vals = [rows_by_id[cid].get(field, 0.0) for cid in ids]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _kendalls_tau(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    pairs = concordant + discordant
    return round((concordant - discordant) / pairs, 3) if pairs > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", default=Path("docs/ablation_report.json"), type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    cache_path = args.cache_dir / "candidate_features.jsonl"
    jd_cache_path = args.cache_dir / "jd_profile.json"

    print("Loading cache...")
    rows = load_cache(cache_path)
    rows_by_id = {r["candidate_id"]: r for r in rows}
    jd_skill_count = load_jd_skill_count(jd_cache_path)
    print(f"Loaded {len(rows)} candidates. JD skill count: {jd_skill_count}")

    print("Computing 5 tier scores per candidate...")
    scores = {r["candidate_id"]: tier_scores(r, jd_skill_count) for r in rows}

    tiers = ["keyword_only", "skill_graph", "agent_rubric", "retrieval_blend", "full_pipeline"]
    top_sets = {t: top_k_ids(rows, t, scores, args.top_k) for t in tiers}

    report: dict[str, Any] = {"top_k": args.top_k, "population": len(rows), "tiers": {}}

    for t in tiers:
        ids = top_sets[t]
        report["tiers"][t] = {
            "off_domain_title_count": off_domain_count(rows_by_id, ids),
            "self_disclosed_transition_count": transition_count(rows_by_id, ids),
            "avg_honeypot_risk": avg_risk(rows_by_id, ids, "honeypot_risk"),
            "avg_stuffer_risk": avg_risk(rows_by_id, ids, "stuffer_risk"),
        }

    report["consecutive_overlap"] = {}
    for i in range(len(tiers) - 1):
        a, b = tiers[i], tiers[i + 1]
        report["consecutive_overlap"][f"{a} -> {b}"] = jaccard(top_sets[a], top_sets[b])
    report["keyword_vs_full_overlap"] = jaccard(top_sets["keyword_only"], top_sets["full_pipeline"])

    pool_ids = list(set(top_sets["keyword_only"]) | set(top_sets["full_pipeline"]))
    x = [scores[cid]["keyword_only"] for cid in pool_ids]
    y = [scores[cid]["full_pipeline"] for cid in pool_ids]
    report["keyword_vs_full_tau"] = _kendalls_tau(x, y)

    kw_top10 = top_sets["keyword_only"][:10]
    full_rank = {cid: i + 1 for i, cid in enumerate(top_k_ids(rows, "full_pipeline", scores, len(rows)))}
    flips = sorted(((cid, full_rank.get(cid, len(rows))) for cid in kw_top10), key=lambda x: -x[1])[:3]
    report["biggest_drops_keyword_to_full"] = [
        {"candidate_id": cid, "full_pipeline_rank": rank, "candidate_name": rows_by_id[cid].get("candidate_name")}
        for cid, rank in flips
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== Ablation summary (top-{args.top_k}) ===")
    for t in tiers:
        d = report["tiers"][t]
        print(f"{t:18s} off_domain={d['off_domain_title_count']:3d}  "
              f"transition={d['self_disclosed_transition_count']:3d}  "
              f"honeypot_avg={d['avg_honeypot_risk']:.3f}  stuffer_avg={d['avg_stuffer_risk']:.3f}")
    print(f"\nKeyword-only vs full-pipeline overlap (Jaccard): {report['keyword_vs_full_overlap']}")
    print(f"Kendall's tau: {report['keyword_vs_full_tau']}")
    print("\nBiggest demotions from keyword-only top-10 → full pipeline:")
    for f in report["biggest_drops_keyword_to_full"]:
        print(f"  {f['candidate_name']} ({f['candidate_id']}) → rank {f['full_pipeline_rank']}")
    print(f"\nFull report → {args.output}")


if __name__ == "__main__":
    main()