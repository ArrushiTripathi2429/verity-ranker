"""
Tests for Layer 2 — JD Intelligence Agent.

All tests run in force_fallback=True mode so no API key is needed.
LLM mode is validated by checking the structured_completion contract,
not by making real API calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.ingestion.schemas import JDInput
from ai_hiring_ranker.jd_intelligence.agent import analyse_jd, _run_fallback
from ai_hiring_ranker.jd_intelligence.schemas import (
    AmbiguityFlag,
    EmploymentType,
    HiringProfile,
    SeniorityLevel,
    SkillEntry,
)

SAMPLE_JD_PATH = ROOT / "data" / "sample" / "jd.txt"




def make_jd(text: str) -> JDInput:
    return JDInput(raw_text=text)


def load_sample_jd() -> JDInput:
    from ai_hiring_ranker.ingestion.loader import load_jd
    return load_jd(SAMPLE_JD_PATH)



class TestSkillEntry:
    def test_skill_title_cased(self):
        s = SkillEntry(skill="python", is_required=True, is_preferred=False)
        assert s.skill == "Python"

    def test_skill_strips_whitespace(self):
        s = SkillEntry(skill="  fastapi  ", is_required=True, is_preferred=False)
        assert s.skill == "Fastapi"


class TestHiringProfile:
    def test_required_skill_names(self):
        profile = HiringProfile(
            job_title="ML Engineer",
            required_skills=[SkillEntry(skill="Python", is_required=True, is_preferred=False)],
            preferred_skills=[SkillEntry(skill="Docker", is_required=False, is_preferred=True)],
        )
        assert "Python" in profile.all_required_skill_names
        assert "Python" not in profile.all_preferred_skill_names

    def test_all_skill_names_combines_both(self):
        profile = HiringProfile(
            job_title="ML Engineer",
            required_skills=[SkillEntry(skill="Python", is_required=True, is_preferred=False)],
            preferred_skills=[SkillEntry(skill="Docker", is_required=False, is_preferred=True)],
        )
        assert set(profile.all_skill_names) == {"Python", "Docker"}

    def test_empty_profile_valid(self):
        profile = HiringProfile(job_title="Unknown Role")
        assert profile.seniority == SeniorityLevel.UNKNOWN
        assert profile.employment_type == EmploymentType.UNKNOWN
        assert profile.required_skills == []




class TestFallbackExtraction:
    def test_sample_jd_extracts_python(self):
        jd = load_sample_jd()
        profile = _run_fallback(jd)
        skill_names = [s.skill.lower() for s in profile.required_skills + profile.preferred_skills]
        assert any("python" in n for n in skill_names)

    def test_sample_jd_extracts_fastapi(self):
        jd = load_sample_jd()
        profile = _run_fallback(jd)
        skill_names = [s.skill.lower() for s in profile.required_skills + profile.preferred_skills]
        assert any("fastapi" in n for n in skill_names)

    def test_preferred_skills_separated(self):
        jd = make_jd(
            "We need a Python engineer. Required: Python, FastAPI, Docker. "
            "Preferred: Kubernetes, Spark. The role involves building production APIs "
            "and deploying models to cloud infrastructure."
        )
        profile = _run_fallback(jd)
        preferred_names = [s.skill.lower() for s in profile.preferred_skills]
        # Kubernetes and Spark are in the preferred sentence
        assert any("kubernetes" in n for n in preferred_names)

    def test_seniority_senior_detected(self):
        jd = make_jd(
            "We are hiring a Senior Machine Learning Engineer with 5+ years of experience "
            "to build and deploy production ML systems using Python and FastAPI."
        )
        profile = _run_fallback(jd)
        assert profile.seniority == SeniorityLevel.SENIOR

    def test_seniority_junior_detected(self):
        jd = make_jd(
            "Junior Data Scientist role. You will work with Python and SQL to build "
            "reporting dashboards and support the analytics team."
        )
        profile = _run_fallback(jd)
        assert profile.seniority == SeniorityLevel.JUNIOR

    def test_seniority_unknown_when_not_mentioned(self):
        jd = make_jd(
            "We need a Python developer to build REST APIs with FastAPI and Docker. "
            "The role involves writing tests and maintaining production services."
        )
        profile = _run_fallback(jd)
        assert profile.seniority == SeniorityLevel.UNKNOWN

    def test_employment_type_full_time(self):
        jd = make_jd(
            "Full-time Machine Learning Engineer. Build Python ML systems, "
            "deploy FastAPI services, and maintain production Docker infrastructure."
        )
        profile = _run_fallback(jd)
        assert profile.employment_type == EmploymentType.FULL_TIME

    def test_years_experience_extracted(self):
        jd = make_jd(
            "Requires 3+ years of experience building Python ML services with FastAPI "
            "and Docker. Must deploy and maintain production APIs."
        )
        profile = _run_fallback(jd)
        assert profile.years_of_experience_min == 3

    def test_responsibilities_extracted(self):
        jd = load_sample_jd()
        profile = _run_fallback(jd)
        assert len(profile.key_responsibilities) > 0

    def test_job_title_extracted(self):
        jd = load_sample_jd()
        profile = _run_fallback(jd)
        assert "Machine Learning Engineer" in profile.job_title

    def test_domain_ml_detected(self):
        jd = load_sample_jd()
        profile = _run_fallback(jd)
        assert profile.domain == "Machine Learning"

    def test_ambiguity_etc_flagged(self):
        jd = make_jd(
            "Required skills: Python, SQL, Docker, etc. The candidate must build "
            "production-ready APIs with FastAPI and deploy them to cloud."
        )
        profile = _run_fallback(jd)
        phrases = [f.phrase.lower() for f in profile.ambiguity_flags]
        assert any("etc" in p for p in phrases)

    def test_hidden_expectation_production_inferred(self):
        jd = make_jd(
            "Build reliable, production-ready Python services with FastAPI. "
            "Deploy to cloud. Write tests. Maintain documentation."
        )
        profile = _run_fallback(jd)
        descriptions = [h.description.lower() for h in profile.hidden_expectations]
        assert any("production" in d for d in descriptions)

    def test_hidden_expectation_team_communication(self):
        jd = make_jd(
            "Collaborate with cross-functional teams. Build Python ML services "
            "and deploy FastAPI APIs. Work with product stakeholders."
        )
        profile = _run_fallback(jd)
        descriptions = [h.description.lower() for h in profile.hidden_expectations]
        assert any("communication" in d for d in descriptions)

    def test_no_skills_invented(self):
        jd = make_jd(
            "We need someone who is passionate, hardworking, and a team player. "
            "Must be willing to learn and grow. Remote work available worldwide."
        )
        profile = _run_fallback(jd)
        # No real tech skills — should extract nothing
        assert len(profile.required_skills) == 0
        assert len(profile.preferred_skills) == 0


# ---------------------------------------------------------------------------
# analyse_jd() integration tests (fallback mode)
# ---------------------------------------------------------------------------


class TestAnalyseJD:
    def test_returns_hiring_profile(self):
        jd = load_sample_jd()
        profile = analyse_jd(jd, force_fallback=True)
        assert isinstance(profile, HiringProfile)

    def test_force_fallback_works_without_key(self):
        jd = make_jd(
            "Senior Python Engineer. Build and deploy FastAPI services with Docker. "
            "5+ years of Python experience required. Full-time role."
        )
        profile = analyse_jd(jd, force_fallback=True)
        assert profile.job_title != ""
        assert profile.seniority == SeniorityLevel.SENIOR

    def test_required_and_preferred_disjoint(self):
        jd = load_sample_jd()
        profile = analyse_jd(jd, force_fallback=True)
        req_names = set(profile.all_required_skill_names)
        pref_names = set(profile.all_preferred_skill_names)
        # A skill should not appear in both lists
        assert req_names.isdisjoint(pref_names), (
            f"Skills appear in both required and preferred: {req_names & pref_names}"
        )

    def test_output_is_serialisable(self):
        jd = load_sample_jd()
        profile = analyse_jd(jd, force_fallback=True)
        data = profile.model_dump()
        assert isinstance(data, dict)
        assert "required_skills" in data
        assert "hidden_expectations" in data
