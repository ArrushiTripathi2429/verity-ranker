"""
Hackathon-native Fairness + Proxy Audit and Rank Stability Test.

Operates directly on cache/candidate_features.jsonl (final_score + dimensions)
and the raw candidates.jsonl (education, career_history, profile) — no
CandidateProfile/RerankResult objects required.

Dev-time diagnostic only. NOT part of rank.py / the timed sandbox path.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

WARNING_RATIO = 2.0
HIGH_RATIO = 3.5
STABLE_RANK_RANGE = 1
WEIGHT_JITTER = 0.05
SCORE_JITTER = 0.03

DEFAULT_WEIGHTS = {
    "skill_fit": 0.30, "experience_depth": 0.20,
    "seniority_match": 0.15, "domain_match": 0.15,
    "career_growth": 0.10, "proof_strength": 0.10,
}


# ── Attribute extractors (raw record → proxy value) ─────────────────────────

def extract_institution(record: dict) -> Optional[str]:
    edu = record.get("education") or []
    for e in edu:
        if e.get("institution"):
            return e["institution"].strip().lower()
    return None


def extract_institution_tier(record: dict) -> Optional[str]:
    """Coarse prestige proxy — far more statistically meaningful at 100K
    scale than raw institution name (thousands of unique names = noise)."""
    edu = record.get("education") or []
    for e in edu:
        if e.get("tier"):
            return e["tier"]
    return None


def extract_graduation_year(record: dict) -> Optional[int]:
    edu = record.get("education") or []
    years = [e.get("end_year") for e in edu if e.get("end_year")]
    return max(years) if years else None


def name_gender_signal(record: dict) -> Optional[str]:
    name = ((record.get("profile") or {}).get("anonymized_name") or "").strip()
    if not name:
        return None
    first = name.split()[0]
    if len(first) <= 4:
        return "short_name"
    if len(first) <= 7:
        return "medium_name"
    return "long_name"


def has_career_gap(record: dict, min_gap_days: int = 730) -> bool:
    roles = sorted(
        [r for r in record.get("career_history", []) if r.get("start_date")],
        key=lambda r: r["start_date"],
    )
    for i in range(len(roles) - 1):
        end = roles[i].get("end_date")
        nxt_start = roles[i + 1].get("start_date")
        if not end or not nxt_start:
            continue
        try:
            gap_days = (
                datetime.fromisoformat(nxt_start) - datetime.fromisoformat(end)
            ).days
        except ValueError:
            continue
        if gap_days >= min_gap_days:
            return True
    return False


# ── Kendall's tau (pure, unchanged from generic auditor) ────────────────────

def _kendalls_tau(x: list[float], y: list[float]) -> float:
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
    return (concordant - discordant) / pairs if pairs > 0 else 0.0


# ── Top-k impact ratio (generic over any attribute map) ─────────────────────

def compute_top_k_ratios(
    attr_map: dict[str, Optional[str]],
    ranked_ids: list[str],
    attribute: str,
    ks: tuple[int, ...] = (5, 10, 20, 50, 100),
) -> list[dict[str, Any]]:
    total = len(ranked_ids)
    if total == 0:
        return []
    all_vals = [attr_map.get(cid) for cid in ranked_ids]
    baseline_counter = Counter(v for v in all_vals if v)

    flags: list[dict[str, Any]] = []
    for k in ks:
        if total < k:
            continue
        top_vals = [attr_map.get(cid) for cid in ranked_ids[:k]]
        top_counter = Counter(v for v in top_vals if v)
        for val, top_count in top_counter.items():
            top_ratio = top_count / k
            baseline_ratio = baseline_counter.get(val, 0) / total
            impact = top_ratio / baseline_ratio if baseline_ratio > 0 else 0.0
            if impact >= WARNING_RATIO:
                flags.append({
                    "attribute": attribute, "value": val, "k": k,
                    "top_k_ratio": round(top_ratio, 3),
                    "baseline_ratio": round(baseline_ratio, 3),
                    "impact_ratio": round(impact, 2),
                    "severity": "high" if impact >= HIGH_RATIO else "warning",
                    "affected_ids": [
                        cid for cid in ranked_ids[:k] if attr_map.get(cid) == val
                    ],
                })
    return flags


# ── Loaders ───────────────────────────────────────────────────────────────

def load_full_ranking(cache_path: Path | str) -> list[tuple[str, float]]:
    """Sort ALL cached candidates by final_score, same tie-break as ranker.py."""
    rows = []
    with Path(cache_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows.append((d["candidate_id"], float(d.get("final_score", 0.0))))
    rows.sort(key=lambda x: (-round(x[1], 2), x[0]))
    return rows


def load_dimensions_for(cache_path: Path | str, ids: set[str]) -> dict[str, dict]:
    out = {}
    with Path(cache_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["candidate_id"] in ids:
                out[d["candidate_id"]] = d
    return out


def load_raw_records(candidates_path: Path | str, ids: set[str]) -> dict[str, dict]:
    out = {}
    with Path(candidates_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["candidate_id"] in ids:
                out[r["candidate_id"]] = r
    return out


# ── Audit 1: Fairness + Proxy Audit ──────────────────────────────────────

def run_fairness_audit(
    candidates_path: Path | str,
    cache_path: Path | str,
    top_ks: tuple[int, ...] = (5, 10, 20, 50, 100),
) -> dict[str, Any]:
    full_ranking = load_full_ranking(cache_path)
    ranked_ids = [cid for cid, _ in full_ranking]
    all_ids = set(ranked_ids)

    print(f"Loading raw records for {len(all_ids)} candidates (one streaming pass)...")
    raw = load_raw_records(candidates_path, all_ids)

    tier_map = {cid: extract_institution_tier(raw[cid]) for cid in ranked_ids if cid in raw}
    inst_map = {cid: extract_institution(raw[cid]) for cid in ranked_ids if cid in raw}
    grad_map = {cid: extract_graduation_year(raw[cid]) for cid in ranked_ids if cid in raw}
    gender_map = {cid: name_gender_signal(raw[cid]) for cid in ranked_ids if cid in raw}
    gap_map = {cid: has_career_gap(raw[cid]) for cid in ranked_ids if cid in raw}

    flags: list[dict[str, Any]] = []
    flags += compute_top_k_ratios(tier_map, ranked_ids, "institution_tier", top_ks)
    flags += compute_top_k_ratios(inst_map, ranked_ids, "institution_name", top_ks)
    flags += compute_top_k_ratios(gender_map, ranked_ids, "name_length_bucket", top_ks)

    # Graduation year vs rank correlation (top-1000 window — recency proxy)
    window = ranked_ids[:1000]
    pairs = [
        (grad_map[cid], i + 1) for i, cid in enumerate(window) if grad_map.get(cid)
    ]
    tau = None
    if len(pairs) >= 5:
        years = [p[0] for p in pairs]
        ranks = [p[1] for p in pairs]
        tau = round(_kendalls_tau(years, ranks), 3)
        if abs(tau) >= 0.35:
            flags.append({
                "attribute": "graduation_year", "value": None, "k": len(window),
                "tau": tau,
                "severity": "high" if abs(tau) >= 0.55 else "warning",
                "note": (
                    "recent graduates rank higher" if tau < 0 else
                    "older graduates rank higher"
                ),
            })

    # Career gap penalty — top-100 window
    top100 = ranked_ids[:100]
    gap_in_top = sum(1 for cid in top100 if gap_map.get(cid))
    gap_in_full = sum(1 for cid in ranked_ids if gap_map.get(cid))
    gap_top_ratio = gap_in_top / 100
    gap_full_ratio = gap_in_full / len(ranked_ids)
    gap_impact = gap_top_ratio / gap_full_ratio if gap_full_ratio > 0 else 0.0
    if gap_impact < 0.5 and gap_full_ratio > 0.02:
        flags.append({
            "attribute": "career_gap", "value": "has_gap_2y+", "k": 100,
            "top_k_ratio": round(gap_top_ratio, 3),
            "baseline_ratio": round(gap_full_ratio, 3),
            "impact_ratio": round(gap_impact, 2),
            "severity": "warning",
            "note": "Candidates with 2y+ career gaps are under-represented in top-100 relative to the full pool.",
        })

    overall = "high" if any(f["severity"] == "high" for f in flags) else \
              "warning" if flags else "info"

    return {
        "overall_risk_level": overall,
        "flags": flags,
        "graduation_year_tau": tau,
        "population_size": len(ranked_ids),
    }


# ── Audit 2: Rank Stability Test ─────────────────────────────────────────

def _jitter_weights(weights: dict[str, float], magnitude: float, rng: random.Random) -> dict[str, float]:
    jittered = {k: max(0.01, v + rng.uniform(-magnitude, magnitude)) for k, v in weights.items()}
    total = sum(jittered.values())
    return {k: v / total for k, v in jittered.items()}


def _jitter_dims(dims: dict[str, float], magnitude: float, rng: random.Random) -> dict[str, float]:
    return {k: min(1.0, max(0.0, v + rng.gauss(0, magnitude))) for k, v in dims.items()}


def _weighted_score(dims: dict[str, float], weights: dict[str, float]) -> float:
    return round(sum(dims.get(k, 0.0) * w for k, w in weights.items()) * 100.0, 2)


def run_stability_audit(
    cache_path: Path | str,
    *,
    pool_size: int = 300,
    top_k: int = 100,
    n_runs: int = 5,
    method: str = "weight_jitter",
    weights: Optional[dict[str, float]] = None,
    seed: int = 42,
) -> dict[str, Any]:
    weights = weights or DEFAULT_WEIGHTS
    full_ranking = load_full_ranking(cache_path)
    pool_ids = [cid for cid, _ in full_ranking[:pool_size]]
    cache = load_dimensions_for(cache_path, set(pool_ids))

    base_rank = {cid: i + 1 for i, cid in enumerate(pool_ids)}
    rank_obs: dict[str, list[int]] = {cid: [] for cid in pool_ids}
    score_obs: dict[str, list[float]] = {cid: [] for cid in pool_ids}
    rng = random.Random(seed)

    for _ in range(n_runs):
        if method == "weight_jitter":
            w = _jitter_weights(weights, WEIGHT_JITTER, rng)
            run_scores = {cid: _weighted_score(cache[cid]["dimensions"], w) for cid in pool_ids}
        else:
            run_scores = {
                cid: _weighted_score(_jitter_dims(cache[cid]["dimensions"], SCORE_JITTER, rng), weights)
                for cid in pool_ids
            }
        reranked = sorted(run_scores.items(), key=lambda x: (-x[1], x[0]))
        for new_rank, (cid, score) in enumerate(reranked, 1):
            rank_obs[cid].append(new_rank)
            score_obs[cid].append(score)

    per_candidate = []
    unstable = []
    for cid in pool_ids[:top_k]:
        obs = rank_obs[cid]
        min_r, max_r = min(obs), max(obs)
        mean_s = sum(score_obs[cid]) / len(score_obs[cid])
        std_s = math.sqrt(sum((s - mean_s) ** 2 for s in score_obs[cid]) / len(score_obs[cid]))
        is_stable = (max_r - min_r) <= STABLE_RANK_RANGE
        per_candidate.append({
            "candidate_id": cid, "base_rank": base_rank[cid],
            "min_rank_observed": min_r, "max_rank_observed": max_r,
            "score_std": round(std_s, 3), "is_stable": is_stable,
        })
        if not is_stable:
            unstable.append(cid)

    return {
        "perturbation_runs": n_runs, "method": method, "pool_size": pool_size,
        "evaluated_top_k": top_k,
        "stable_count": top_k - len(unstable),
        "unstable_count": len(unstable),
        "unstable_ids": unstable,
        "candidate_stability": per_candidate,
    }