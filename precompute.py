#!/usr/bin/env python3
"""
Precompute hackathon features (internet allowed, no time limit).

Runs JD intelligence + offline candidate feature extraction once and writes:
  <cache-dir>/jd_profile.json
  <cache-dir>/candidate_features.jsonl
  <cache-dir>/manifest.json

No GitHub API calls — uses github_activity_score from the dataset only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.hackathon.dataset import iter_candidates  # noqa: E402
from ai_hiring_ranker.hackathon.features import build_features  # noqa: E402
from ai_hiring_ranker.hackathon.schemas import PrecomputeManifest  # noqa: E402
from ai_hiring_ranker.ingestion.schemas import JDInput  # noqa: E402
from ai_hiring_ranker.jd_intelligence.agent import analyse_jd  # noqa: E402

logger = logging.getLogger(__name__)


def _has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY"))


def run_precompute(
    jd_path: Path,
    candidates_path: Path,
    cache_dir: Path,
    *,
    force_fallback: bool = False,
    limit: int | None = None,
) -> PrecomputeManifest:
    cache_dir.mkdir(parents=True, exist_ok=True)
    jd_cache = cache_dir / "jd_profile.json"
    feature_cache = cache_dir / "candidate_features.jsonl"

    jd_text = jd_path.read_text(encoding="utf-8")
    jd_input = JDInput(raw_text=jd_text, source_path=str(jd_path))
    use_fallback = force_fallback or not _has_api_key()
    hiring_profile = analyse_jd(jd_input, force_fallback=use_fallback)

    jd_cache.write_text(
        hiring_profile.model_dump_json(indent=2),
        encoding="utf-8",
    )

    count = 0
    t0 = time.perf_counter()
    with feature_cache.open("w", encoding="utf-8") as out:
        for line_no, record in iter_candidates(candidates_path):
            if limit is not None and count >= limit:
                break
            features = build_features(
                record,
                hiring_profile,
                line_no=line_no,
                force_fallback=True,  # extraction always offline-safe
            )
            out.write(json.dumps(features, ensure_ascii=False) + "\n")
            count += 1
            if count % 5000 == 0:
                logger.info("Precomputed %d candidates...", count)

    manifest = PrecomputeManifest(
        jd_path=str(jd_path.resolve()),
        candidates_path=str(candidates_path.resolve()),
        candidate_count=count,
        force_fallback=use_fallback,
        used_llm=not use_fallback,
        job_title=hiring_profile.job_title,
        cache_file=feature_cache.name,
        jd_cache_file=jd_cache.name,
    )
    (cache_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Precompute complete: %d candidates in %.1fs → %s",
        count,
        elapsed,
        cache_dir,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute hackathon candidate features.")
    parser.add_argument("--jd", required=True, type=Path, help="Path to job description text file")
    parser.add_argument("--candidates", required=True, type=Path, help="Path to candidates.jsonl")
    parser.add_argument("--cache-dir", default=Path("cache"), type=Path, help="Output cache directory")
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Skip LLM for JD analysis (fully offline)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N candidates")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    run_precompute(
        jd_path=args.jd,
        candidates_path=args.candidates,
        cache_dir=args.cache_dir,
        force_fallback=args.force_fallback,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
