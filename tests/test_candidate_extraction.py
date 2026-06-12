"""
Tests for Layer 4 — Candidate Profile Extraction.

All tests run in force_fallback=True (no API key required).
Covers schema validation, individual rule-based extractors,
signal computation, and full end-to-end extraction from sample data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.ingestion.schemas import CandidateInput
from ai_hiring_ranker.ingestion.loader import ingest
from ai_hiring_ranker.candidate_extraction.extractor import (
    extract_candidate_profile,
    extract_all_candidates,
    _run_fallback,
    _extract_skills,
    _extract_achievements,
    _extract_projects,
    _compute_seniority_signal,
    _compute_leadership_signal,
    _compute_production_signal,
    _sentences,
)
from ai_hiring_ranker.candidate_extraction.schemas import (
    Achievement,
    CandidateProfile,
    SkillClaim,
    SkillConfidence,
    CareerRole,
    ProjectEntry,
)

SAMPLE_CANDIDATES = ROOT / "data" / "sample" / "candidates"
SAMPLE_JD         = ROOT / "data" / "sample" / "jd.txt"




def make_candidate(text: str, cid: str = "TEST001") -> CandidateInput:
    return CandidateInput(candidate_id=cid, raw_text=text)


def load_sample_candidates() -> list[CandidateInput]:
    result = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
    return result.candidates



class TestSkillClaim:
    def test_skill_title_cased(self):
        sc = SkillClaim(skill="python", confidence=SkillConfidence.EXPLICIT)
        assert sc.skill == "Python"

    def test_default_confidence(self):
        sc = SkillClaim(skill="Docker")
        assert sc.confidence == SkillConfidence.EXPLICIT

    def test_evidence_snippets_default_empty(self):
        sc = SkillClaim(skill="SQL")
        assert sc.evidence_snippets == []


class TestCareerRole:
    def test_duration_auto_computed(self):
        role = CareerRole(title="Engineer", start_year=2019, end_year=2022)
        assert role.duration_years == 3.0

    def test_duration_not_overwritten_if_set(self):
        role = CareerRole(title="Engineer", start_year=2019, end_year=2022, duration_years=2.5)
        assert role.duration_years == 2.5

    def test_missing_years_gives_none_duration(self):
        role = CareerRole(title="Intern")
        assert role.duration_years is None


class TestCandidateProfile:
    def test_skill_names_property(self):
        profile = CandidateProfile(
            candidate_id="X",
            skills=[
                SkillClaim(skill="Python", confidence=SkillConfidence.EXPLICIT),
                SkillClaim(skill="Docker", confidence=SkillConfidence.INFERRED),
            ],
        )
        assert "Python" in profile.skill_names
        assert "Docker" in profile.skill_names

    def test_explicit_skill_names_filter(self):
        profile = CandidateProfile(
            candidate_id="X",
            skills=[
                SkillClaim(skill="Python", confidence=SkillConfidence.EXPLICIT),
                SkillClaim(skill="Docker", confidence=SkillConfidence.INFERRED),
            ],
        )
        assert "Python" in profile.explicit_skill_names
        assert "Docker" not in profile.explicit_skill_names

    def test_has_production_evidence_false_by_default(self):
        profile = CandidateProfile(candidate_id="X")
        assert not profile.has_production_evidence

    def test_has_production_evidence_true_when_signal_high(self):
        profile = CandidateProfile(candidate_id="X", production_signal=0.8)
        assert profile.has_production_evidence

    def test_has_leadership_evidence(self):
        profile = CandidateProfile(candidate_id="X", leadership_signal=0.6)
        assert profile.has_leadership_evidence





class TestExtractSkills:
    def test_python_detected(self):
        text = "I have 3 years of Python experience building ML services."
        sents = _sentences(text)
        skills = _extract_skills(text, sents)
        names = [s.skill.lower() for s in skills]
        assert "python" in names

    def test_negated_skill_excluded(self):
        text = (
            "I have no experience with Docker. "
            "I am skilled in Python and FastAPI for building REST APIs."
        )
        sents = _sentences(text)
        skills = _extract_skills(text, sents)
        names = [s.skill.lower() for s in skills]
        # Docker appears in a negation sentence — its evidence should be empty
        docker = next((s for s in skills if "docker" in s.skill.lower()), None)
        if docker:
            assert len(docker.evidence_snippets) == 0

    def test_explicit_confidence_for_skills_line(self):
        text = "Skills: Python, FastAPI, Docker\nI build ML services."
        sents = _sentences(text)
        skills = _extract_skills(text, sents)
        python = next((s for s in skills if "python" in s.skill.lower()), None)
        assert python is not None
        assert python.confidence == SkillConfidence.EXPLICIT

    def test_evidence_snippets_non_empty(self):
        text = "I built a FastAPI inference API and deployed it to Docker. FastAPI was core."
        sents = _sentences(text)
        skills = _extract_skills(text, sents)
        fastapi = next((s for s in skills if "fastapi" in s.skill.lower()), None)
        assert fastapi is not None
        assert len(fastapi.evidence_snippets) > 0

    def test_years_extracted_from_snippet(self):
        text = "I have 4 years of Python experience in production ML systems."
        sents = _sentences(text)
        skills = _extract_skills(text, sents)
        python = next((s for s in skills if "python" in s.skill.lower()), None)
        assert python is not None
        assert python.years_of_experience == 4.0


class TestExtractAchievements:
    def test_achievement_with_metric(self):
        sents = ["Reduced inference latency by 40% by switching to ONNX runtime."]
        achievements = _extract_achievements(sents)
        assert len(achievements) == 1
        assert achievements[0].has_metric is True
        assert "40%" in achievements[0].metric_snippet

    def test_achievement_without_metric(self):
        sents = ["Built a data pipeline that automated weekly report generation."]
        achievements = _extract_achievements(sents)
        assert len(achievements) == 1
        assert achievements[0].has_metric is False

    def test_no_achievement_keywords(self):
        sents = ["I am a team player who loves learning new technologies."]
        achievements = _extract_achievements(sents)
        assert len(achievements) == 0


class TestExtractProjects:
    def test_project_detected(self):
        sents = ["Built a retrieval system using FAISS and sentence transformers for semantic search."]
        projects = _extract_projects(sents)
        assert len(projects) == 1

    def test_production_flag(self):
        sents = ["Deployed a FastAPI inference service to production serving 10k requests/day."]
        projects = _extract_projects(sents)
        assert len(projects) == 1
        assert projects[0].is_production is True

    def test_metrics_flag(self):
        sents = ["Created a model evaluation pipeline that improved accuracy by 15%."]
        projects = _extract_projects(sents)
        assert len(projects) == 1
        assert projects[0].has_metrics is True

    def test_skills_extracted_from_project(self):
        sents = ["Developed a Docker-based FastAPI service with Python for model serving."]
        projects = _extract_projects(sents)
        assert len(projects) == 1
        skill_names = [s.lower() for s in projects[0].skills_used]
        assert any("docker" in s for s in skill_names)


class TestSignals:
    def test_senior_seniority_signal(self):
        text = "Senior Machine Learning Engineer with 6 years of experience."
        assert _compute_seniority_signal(text) >= 0.7

    def test_junior_seniority_signal(self):
        text = "Junior Data Scientist with 1 year of experience in Python."
        assert _compute_seniority_signal(text) <= 0.35

    def test_leadership_signal_present(self):
        text = "Led a team of 5 engineers. Mentored junior developers. Drove the architecture."
        sig = _compute_leadership_signal(text)
        assert sig >= 0.6

    def test_leadership_signal_absent(self):
        text = "I write Python code and analyse data using pandas."
        sig = _compute_leadership_signal(text)
        assert sig == 0.0

    def test_production_signal_present(self):
        sents = [
            "Deployed the FastAPI service to production.",
            "The service ran in production serving 5k users.",
            "Released version 2 after load testing.",
        ]
        sig = _compute_production_signal(sents)
        assert sig >= 0.6

    def test_production_signal_absent(self):
        sents = ["Explored machine learning algorithms in Jupyter notebooks."]
        sig = _compute_production_signal(sents)
        assert sig == 0.0





class TestFullExtraction:
    def test_c001_python_extracted(self):
        cands = load_sample_candidates()
        c001 = next(c for c in cands if c.candidate_id == "C001")
        profile = extract_candidate_profile(c001, force_fallback=True)
        names = [s.skill.lower() for s in profile.skills]
        assert "python" in names

    def test_c001_fastapi_extracted(self):
        cands = load_sample_candidates()
        c001 = next(c for c in cands if c.candidate_id == "C001")
        profile = extract_candidate_profile(c001, force_fallback=True)
        names = [s.skill.lower() for s in profile.skills]
        assert "fastapi" in names

    def test_c001_name_extracted(self):
        cands = load_sample_candidates()
        c001 = next(c for c in cands if c.candidate_id == "C001")
        profile = extract_candidate_profile(c001, force_fallback=True)
        assert "Alex" in profile.name or "Rivera" in profile.name

    def test_c001_has_projects(self):
        cands = load_sample_candidates()
        c001 = next(c for c in cands if c.candidate_id == "C001")
        profile = extract_candidate_profile(c001, force_fallback=True)
        assert profile.total_projects > 0

    def test_c001_production_signal(self):
        cands = load_sample_candidates()
        c001 = next(c for c in cands if c.candidate_id == "C001")
        profile = extract_candidate_profile(c001, force_fallback=True)
        # C001 has deployment/production language
        assert profile.production_signal > 0.0

    def test_c003_junior_seniority(self):
        cands = load_sample_candidates()
        c003 = next(c for c in cands if c.candidate_id == "C003")
        profile = extract_candidate_profile(c003, force_fallback=True)
        assert profile.seniority_signal <= 0.55

    def test_c002_has_sql(self):
        cands = load_sample_candidates()
        c002 = next(c for c in cands if c.candidate_id == "C002")
        profile = extract_candidate_profile(c002, force_fallback=True)
        names = [s.skill.lower() for s in profile.skills]
        assert "sql" in names

    def test_all_profiles_have_id(self):
        cands = load_sample_candidates()
        profiles = extract_all_candidates(cands, force_fallback=True)
        for p in profiles:
            assert p.candidate_id != ""

    def test_all_profiles_serialisable(self):
        cands = load_sample_candidates()
        profiles = extract_all_candidates(cands, force_fallback=True)
        for p in profiles:
            data = p.model_dump()
            assert "skills" in data
            assert "candidate_id" in data

    def test_extract_all_returns_same_count(self):
        cands = load_sample_candidates()
        profiles = extract_all_candidates(cands, force_fallback=True)
        assert len(profiles) == len(cands)




class TestEndToEnd:
    def test_full_pipeline_l1_to_l4(self):
        from ai_hiring_ranker.ingestion.loader import ingest
        from ai_hiring_ranker.jd_intelligence.agent import analyse_jd
        from ai_hiring_ranker.hyde.generator import generate_hyde_profiles
        from ai_hiring_ranker.candidate_extraction.extractor import extract_all_candidates

        result       = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        hiring_prof  = analyse_jd(result.jd, force_fallback=True)
        hyde_result  = generate_hyde_profiles(hiring_prof, force_fallback=True)
        profiles     = extract_all_candidates(result.candidates, force_fallback=True)

        assert len(profiles) == result.candidate_count
        assert all(isinstance(p, CandidateProfile) for p in profiles)
        assert hyde_result.job_title != ""
        # Every profile has at least one skill
        assert all(len(p.skills) > 0 for p in profiles)
