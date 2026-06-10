# Verity Ranker V1

V1 is the minimum working candidate ranking pipeline.

It is intentionally limited. It does not claim full agentic evaluation, proof-of-work verification, graph reasoning, or fairness guarantees. Its job is to prove that the project can:

- read a job description
- read candidate profiles/resumes
- extract basic role and candidate signals
- rank candidates with a deterministic scoring rubric
- produce a valid ranked output file
- generate a small audit report for review

## Why V1 Exists

Most advanced hiring-ranker ideas fail if the basic output file is wrong. V1 creates the clean baseline before adding claim verification, graph retrieval, multi-agent scoring, and reranking.

## Run Sample

From this folder:

```bash
python run_v1.py run --jd data/sample/jd.txt --candidates data/sample/candidates --output outputs/final/ranked_output.csv --report outputs/final/audit_report.json
python run_v1.py validate --file outputs/final/ranked_output.csv
```

## Current Scope

Included:

- basic JD parsing
- basic candidate parsing
- skill extraction through a controlled skill vocabulary
- deterministic weighted scoring
- ranked CSV output
- JSON audit report
- output validation

Not included:

- GitHub proof-of-work verification
- evidence ledger with source IDs
- skill/role knowledge graph
- HyDE retrieval
- multi-agent committee
- listwise reranking
- fairness/proxy audit
- rank stability audit

These are intentionally left for later versions.

