"""Tests for hackathon precompute + offline rank path."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.hackathon.ranker import load_feature_cache, rank_candidates  # noqa: E402
from scripts.validate_submission import validate_submission  # noqa: E402

JD = ROOT / "data" / "hackathon" / "jd.txt"
CANDIDATES = ROOT / "data" / "hackathon" / "candidates.jsonl"
CACHE = ROOT / "outputs" / "hackathon_cache_test"
OUTPUT = ROOT / "outputs" / "hackathon_rank_test.csv"


@pytest.fixture(scope="module", autouse=True)
def ensure_data():
  gen = ROOT / "scripts" / "generate_hackathon_data.py"
  if not CANDIDATES.exists():
    subprocess.run(
      [sys.executable, str(gen), "--output", str(CANDIDATES), "--count", "200"],
      check=True,
      cwd=ROOT,
    )


def test_precompute_and_rank_pipeline():
  subprocess.run(
    [
      sys.executable,
      str(ROOT / "precompute.py"),
      "--jd",
      str(JD),
      "--candidates",
      str(CANDIDATES),
      "--cache-dir",
      str(CACHE),
      "--force-fallback",
    ],
    check=True,
    cwd=ROOT,
  )

  t0 = time.perf_counter()
  subprocess.run(
    [
      sys.executable,
      str(ROOT / "rank.py"),
      "--jd",
      str(JD),
      "--candidates",
      str(CANDIDATES),
      "--cache-dir",
      str(CACHE),
      "--output",
      str(OUTPUT),
    ],
    check=True,
    cwd=ROOT,
  )
  elapsed = time.perf_counter() - t0
  assert elapsed < 60, f"rank.py too slow: {elapsed:.1f}s"

  errors = validate_submission(OUTPUT)
  assert errors == [], "\n".join(errors)


def test_honeypots_not_in_top_10():
  features = load_feature_cache(CACHE / "candidate_features.jsonl")
  rows = rank_candidates(features, job_title="Machine Learning Engineer", top_k=100)
  top10_ids = {r.candidate_id for r in rows[:10]}
  assert "HONEYPOT01" not in top10_ids
  assert "HONEYPOT02" not in top10_ids


def test_submission_csv_format():
  with OUTPUT.open(encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    assert reader.fieldnames == ["candidate_id", "rank", "score", "reasoning"]
    rows = list(reader)
  assert len(rows) == 100
