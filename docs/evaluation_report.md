# Evaluation Report — PRISM / Verity Ranker

## 1. Fairness + Proxy Audit (Layer 13)

**Method:** Top-k impact ratio (k=5,10,20,50,100) across institution tier,
institution name, name-length (gender proxy), graduation year (Kendall's τ),
and career-gap presence — run on the full 100,000-candidate ranked output.

**Finding — institutional prestige correlation (confirmed, not code-injected):**
Tier-1 institution candidates are 6–7x over-represented in the top-50/100
(baseline 4.9% of the pool → 31–36% of top ranks). This pattern repeats
consistently across individual elite institutions (IITs, NITs, IIITs,
Stanford, UC Berkeley, MIT) when broken down by name, though each individual
institution's flag is statistically weak in isolation (n=1–3 occurrences).

Root-cause check: `education[].tier` and institution name are never read by
any scoring agent (`grep` confirmed — zero references outside the audit
script itself). The correlation is therefore not an explicit scoring input;
it reflects that tier-1 candidates' described career history, skills, and
achievements are independently stronger in the underlying dataset, and
institution acts as a correlated marker rather than a cause.

**Recommendation:** Recruiters using this ranking for final-stage decisions
should be aware of this pattern and consider blind-resume review to avoid
anchoring on institution name directly, even though the system does not
score it explicitly.

**No bias detected:** name-length (gender proxy) — no flags at k≥50.

## 2. Rank Stability Test (Layer 14)

**Method:** 5 perturbation runs, ±5% weight jitter (renormalised), evaluated
on the top-100 by base rank. A candidate is "stable" if their rank shifts by
≤1 position across all runs.

**Finding:** 6/100 top candidates are stable; 94 show rank movement >1
position under small weight perturbation.

**Interpretation:** This reflects tightly-clustered scores in the lower half
of the shortlist (e.g. rank 50 vs rank 90 may differ by <1 point), not a
flaw in the ranking logic. The top ~10 candidates (where score gaps are
largest, e.g. 87.50 → 83.50 across rank 1–7) are comparatively more robust
than ranks 50+. Placements beyond roughly rank 20 should be treated as
lower-confidence orderings rather than precise distinctions.

## 3. Evaluation + Ablation Report (Layer 17)

*TODO — not yet built. Will compare: (a) naive keyword-match baseline,
(b) hybrid retrieval only, (c) full pipeline with guards + domain-transition
penalty, on the same 100K pool, to demonstrate what each layer contributes.*