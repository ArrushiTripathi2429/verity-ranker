#!/usr/bin/env python3
"""
Offline hackathon ranker (sandbox script).

NO network calls. NO GPU. Loads precomputed cache + JD, scores all candidates,
outputs exactly 100 rows:

  candidate_id,rank,score,reasoning

Usage:
  python rank.py --jd data/jd.txt --candidates data/candidates.jsonl --cache-dir cache --output submission/ranked_output.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.hackathon.dataset import iter_candidates  # noqa: E402
from ai_hiring_ranker.hackathon.features import build_features  # noqa: E402
from ai_hiring_ranker.hackathon.ranker import (  # noqa: E402
    load_feature_cache,
    rank_candidates,
    write_submission_csv,
)
from ai_hiring_ranker.ingestion.schemas import JDInput  # noqa: E402
from ai_hiring_ranker.jd_intelligence.agent import analyse_jd  # noqa: E402
from ai_hiring_ranker.jd_intelligence.schemas import HiringProfile  # noqa: E402

logger = logging.getLogger(__name__)


def _load_hiring_profile(jd_path: Path, cache_dir: Path) -> HiringProfile:
    jd_cache = cache_dir / "jd_profile.json"
    if jd_cache.exists():
        return HiringProfile.model_validate_json(jd_cache.read_text(encoding="utf-8"))

    jd_text = jd_path.read_text(encoding="utf-8")
    jd_input = JDInput(raw_text=jd_text, source_path=str(jd_path))
    return analyse_jd(jd_input, force_fallback=True)


def _build_features_offline(
    candidates_path: Path,
    hiring_profile: HiringProfile,
) -> list[dict]:
    rows: list[dict] = []
    for line_no, record in iter_candidates(candidates_path):
        rows.append(
            build_features(record, hiring_profile, line_no=line_no, force_fallback=True)
        )
    return rows


def run_rank(
    jd_path: Path,
    candidates_path: Path,
    cache_dir: Path,
    output_path: Path,
    *,
    top_k: int = 100,
) -> Path:
    t0 = time.perf_counter()
    hiring_profile = _load_hiring_profile(jd_path, cache_dir)

    feature_cache = cache_dir / "candidate_features.jsonl"
    if feature_cache.exists():
        logger.info("Loading precomputed cache: %s", feature_cache)
        features = load_feature_cache(feature_cache)
    else:
        logger.warning(
            "Cache missing — building features inline (slow). Run precompute.py first."
        )
        features = _build_features_offline(candidates_path, hiring_profile)

    rows = rank_candidates(features, job_title=hiring_profile.job_title, top_k=top_k)
    out = write_submission_csv(rows, output_path)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Ranked %d candidates → top %d in %.2fs → %s",
        len(features),
        len(rows),
        elapsed,
        out,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline hackathon ranker (sandbox).")
    parser.add_argument("--jd", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path("cache"), type=Path)
    parser.add_argument("--output", default=Path("submission/ranked_output.csv"), type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    run_rank(
        jd_path=args.jd,
        candidates_path=args.candidates,
        cache_dir=args.cache_dir,
        output_path=args.output,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
