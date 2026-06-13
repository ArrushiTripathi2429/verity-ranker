Overall project
We're building ai-hiring-ranker — an intelligent, evidence-aware candidate ranking system. The architecture has 14 layers total. We've completed 2.

The codebase has two packages living side by side:

src/verity_ranker/ — the original v1 keyword-matching baseline (kept intact, still powers the Streamlit UI)
src/ai_hiring_ranker/ — the new architecture being built layer by layer
Layer 1 — Input Layer 
Purpose: Load, parse, and validate every input before anything else touches it.

Files built:

src/ai_hiring_ranker/ingestion/
    __init__.py
    schemas.py
    parsers.py
    link_extractor.py
    loader.py
What each file does:

schemas.py — Pydantic v2 data models that gate everything entering the pipeline:

FileFormat enum — txt, md, pdf, docx, unknown
VerificationStatus enum — verified, weak, inferred, unsupported, pending (used later in Layer 5)
PortfolioLinks — typed container for github, kaggle, linkedin, and a list of other URLs. Has a has_any property
JDInput — validated JD. Enforces min 50 chars, auto-computes word_count and char_count, records ingested_at timestamp
CandidateInput — validated resume. Enforces min 20 chars, holds candidate_id, portfolio_links, same metadata
parsers.py — multi-format text extraction:

.txt / .md — UTF-8 with latin-1 fallback
.pdf — text-layer extraction via pypdf, warns if no text found (scanned PDF)
.docx — paragraph text + table cell text via python-docx (so skills tables aren't silently dropped)
extract_text(path) — unified dispatcher; raises ValueError for unsupported formats
link_extractor.py — extracts external profile links from resume text using compiled regex:

GitHub profile/repo URLs
Kaggle profile URLs
LinkedIn /in/ URLs
Generic HTTPS URLs for personal sites, Hugging Face, GitLab, etc.
Deduplicates, normalises scheme (adds https:// if missing), excludes already-captured domains from the generic list
loader.py — ingest() is the single public entry point for the whole layer:

load_jd(path) — loads and validates one JD file; fatal if it fails
load_candidate(path) — loads one resume, resolves candidate_id from explicit "Candidate ID:" field or filename stem
ingest(jd_path, candidates_dir) — scans directory, loads everything, returns IngestResult
Supports dual mode: filesystem paths (CLI) and in-memory text tuples (Streamlit uploads)
Per-candidate errors are collected into IngestResult.errors, not raised — one bad file doesn't kill the run
IngestResult.summary() prints a human-readable ingestion report
Tests: 
test_ingestion.py
 — 29 tests, all passing

Layer 2 — JD Intelligence Agent 
Purpose: Convert raw JD text into a structured HiringProfile — the single source of truth about what the role needs. Every layer downstream reads from this.

Files built:

src/ai_hiring_ranker/jd_intelligence/
    __init__.py
    schemas.py
    agent.py

src/ai_hiring_ranker/
    config.py
    llm_provider.py

prompts/
    jd_intelligence.md
What each file does:

schemas.py
 — output data models:

SeniorityLevel enum — intern / junior / mid / senior / staff / principal / lead / manager / unknown
EmploymentType enum — full_time / part_time / contract / freelance / internship / unknown
SkillEntry — a single skill with is_required/is_preferred flags and the JD sentence it came from
HiddenExpectation — an implied requirement (e.g. "build production APIs" implies monitoring knowledge), with inferred_from and a confidence score 0–1
AmbiguityFlag — a vague phrase (e.g. "strong background", "etc.") with a reason and suggested clarification
HiringProfile — the complete structured output with convenience properties: all_required_skill_names, all_preferred_skill_names, all_skill_names
agent.py
 — the agent with two modes:

LLM mode — sends the JD to structured_completion(), gets back guaranteed JSON via response_format: json_object, validates through Pydantic. Auto-selected when OPENAI_API_KEY is present
Fallback mode — pure rule-based extraction, no API key needed. Covers: 25+ skills with aliases, seniority detection (regex patterns ordered most-specific first), employment type detection, responsibility extraction, years-of-experience parsing, ambiguity flagging (etc., "and more", "strong background", etc.), hidden expectation inference (production language → monitoring expectation, collaboration language → communication expectation, etc.)
analyse_jd(jd, force_fallback=False) — public entry point. Auto-selects LLM if key present, falls back gracefully if LLM call fails mid-run
config.py — typed YAML config loader:

Reads 
models.yaml
Pydantic models: LLMConfig, EmbeddingConfig, HyDEConfig, ModelsConfig
@lru_cache so the file is only read once per process
API key injected from OPENAI_API_KEY env var, never from the config file
llm_provider.py — provider abstraction:

chat_completion(system, user) — raw string call to the configured provider
structured_completion(system, user, schema) — calls LLM, extracts JSON from the response (handles markdown fences), validates against a Pydantic model, retries up to 2x with an improved prompt on parse/validation failure
Only OpenAI implemented now; adding Anthropic/local = one new _anthropic_chat() function
jd_intelligence.md
 — system prompt with the exact JSON schema the LLM must return, rules against inventing skills, and instructions for each extraction task

 Layer 3 — HyDE Ideal Candidate Generation is complete.

What was built
schemas.py

CandidateTier enum — minimum, strong, exceptional
IdealCandidateProfile — one hypothetical candidate with profile_text (the embeddable narrative), skills_demonstrated, experience_years, seniority_label, and differentiator (one-liner explaining what separates this tier from the one below)
HyDEResult — container for all three profiles with .minimum, .strong, .exceptional shortcut properties and .all_profile_texts for batch embedding
generator.py

generate_hyde_profiles(hiring_profile) — public entry point; auto-selects LLM vs fallback
LLM mode — uses temperature=0.7 (higher than the JD agent) for richer, more varied profiles; feeds the full HiringProfile including hidden expectations into the prompt
Fallback mode — deterministic template generation driven by the HiringProfile skill lists; minimum tier gets required skills only, strong adds preferred, exceptional adds all with leadership/impact language; experience years scale by seniority level from a lookup table with overrides from the JD's explicit years_of_experience_min
hyde_generation.md
 — detailed system prompt that defines exactly what each tier should and shouldn't contain, with the required JSON schema

Why HyDE matters (the key insight)
Without HyDE, you'd embed the JD text and search for similar resumes. The problem is JDs and resumes don't share the same vocabulary or style — a JD says "we need Python expertise" while a resume says "3 years building production services in Python". The vector similarity is poor.

With HyDE, you generate synthetic text that looks like a resume, embed that, and search. Now you're searching resume-space with resume-space queries — much higher recall.


Layer 4 — Candidate Profile Extraction is complete.

What was built
schemas.py
 — 8 data models

Model	Purpose
SkillConfidence	explicit / inferred / weak — how confidently a skill was extracted
SkillClaim	Skill + evidence snippets (verbatim sentences) + years of experience + last used year. Fed directly into Layer 5 for verification
ProjectEntry	One project with is_production and has_metrics flags + skills used
Achievement	Concrete accomplishment with has_metric + exact metric_snippet (e.g. "40% latency reduction")
CareerRole	One timeline entry — title, company, start/end years, auto-computed duration_years
EducationEntry	Degree, field, institution, graduation year, DegreeLevel enum
CandidateProfile	The complete structured output — everything downstream reads from this
CandidateProfile computed signals (0–1):

seniority_signal — keyword patterns: senior/lead/principal → 0.8–1.0, junior/intern → 0.1–0.25
leadership_signal — counts leadership verbs: led, owned, drove, mentored, architected
production_signal — counts production-language sentences: deployed, shipped, served, released
achievement_signal — volume of achievements + bonus for metric-backed ones
career_growth_signal — number of roles + title progression to senior/lead
extractor.py
 — the extraction engine

_extract_skills() — matches against 30+ skill aliases, skips negation sentences, classifies confidence, collects up to 3 evidence snippets per skill, extracts years and last-used year from context
_extract_projects() — detects build/create/deploy verbs, flags production and metric mentions
_extract_achievements() — detects achievement verbs, captures metric phrases with regex
_extract_career_timeline() — regex for role title + company name patterns
extract_candidate_profile() — auto-selects LLM vs fallback; LLM fails → fallback automatically
extract_all_candidates() — batch processing; per-candidate errors are logged and skipped, not raised
candidate_extraction.md
 — full system prompt with schema, signal definitions, and strict rules against inventing facts


 JD file + Resume files
  ↓ Layer 1: ingest()
JDInput + [CandidateInput]
  ↓ Layer 2: analyse_jd()
HiringProfile (required/preferred skills, seniority, hidden expectations)
  ↓ Layer 3: generate_hyde_profiles()
HyDEResult (3 ideal profiles for retrieval anchoring)
  ↓ Layer 4: extract_all_candidates()
[CandidateProfile] (skills+evidence, projects, achievements, timeline, signals)
