# Hackathon Submission — Verity Ranker

## Two-script design (required for sandbox)

| Script | When to run | Internet | Purpose |
|--------|-------------|----------|---------|
| `precompute.py` | Once, before submission | Optional (LLM for JD only) | Build feature cache from full dataset |
| `rank.py` | Judge sandbox / re-run | **NO** | Load cache → top 100 CSV in <5 min |

## One-command reproduction

```bash
pip install -r requirements.txt
python scripts/generate_hackathon_data.py   # local test data only
python precompute.py --jd data/hackathon/jd.txt --candidates data/hackathon/candidates.jsonl --cache-dir cache --force-fallback
python rank.py --jd data/hackathon/jd.txt --candidates data/hackathon/candidates.jsonl --cache-dir cache --output submission/ranked_output.csv
python scripts/validate_submission.py submission/ranked_output.csv
```

Replace `data/hackathon/candidates.jsonl` with the official 100K file for the real submission.

## Output format

```csv
candidate_id,rank,score,reasoning
C00142,1,87.40,"Strong fit for Machine Learning Engineer: covers required skills Python, FastAPI ..."
```

Exactly **100 rows**. Scores **non-increasing** by rank. Ties broken by `candidate_id` ascending.

## What moved out of rank.py

- GitHub API verification → **removed**; uses `github_activity_score` from dataset
- LLM calls → **precompute only** (optional); `rank.py` is 100% offline
- Full 15-layer pipeline → feature extraction + rubric scoring in cache

## Guards built into scoring

- **Honeypot down-rank** — impossible claims, title/experience conflicts, skill stuffing
- **Keyword-stuffer down-rank** — buzzword density vs profile depth
- **Engagement down-rank** — `last_active_date`, `recruiter_response_rate`

## Sandbox demo

Streamlit app (`streamlit run app.py`) demonstrates the full V2 pipeline on sample data for the portal link requirement.
