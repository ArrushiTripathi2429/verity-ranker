"""
Tests for Layer 3 — HyDE Ideal Candidate Generation.

All tests run in force_fallback=True mode — no API key required.
LLM mode is covered by integration with the llm_provider contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.ingestion.loader import load_jd
from ai_hiring_ranker.jd_intelligence.agent import analyse_jd
from ai_hiring_ranker.jd_intelligence.schemas import (
    HiringProfile,
    SeniorityLevel,
    SkillEntry,
)
from ai_hiring_ranker.hyde.generator import generate_hyde_profiles, _run_fallback
from ai_hiring_ranker.hyde.schemas import CandidateTier, HyDEResult, IdealCandidateProfile

SAMPLE_JD_PATH = ROOT / "data" / "sample" / "jd.txt"





def make_profile(
    title: str = "ML Engineer",
    required: list[str] | None = None,
    preferred: list[str] | None = None,
    seniority: SeniorityLevel = SeniorityLevel.MID,
    years: int | None = 3,
    domain: str | None = "Machine Learning",
) -> HiringProfile:
    req = [SkillEntry(skill=s, is_required=True, is_preferred=False) for s in (required or [])]
    pref = [SkillEntry(skill=s, is_required=False, is_preferred=True) for s in (preferred or [])]
    return HiringProfile(
        job_title=title,
        seniority=seniority,
        required_skills=req,
        preferred_skills=pref,
        years_of_experience_min=years,
        domain=domain,
    )


def sample_profile() -> HiringProfile:
    jd = load_jd(SAMPLE_JD_PATH)
    return analyse_jd(jd, force_fallback=True)





class TestIdealCandidateProfile:
    def test_word_count(self):
        p = IdealCandidateProfile(
            tier=CandidateTier.MINIMUM,
            profile_text="I am an engineer with Python and FastAPI experience. " * 5,
            skills_demonstrated=["Python"],
        )
        assert p.word_count() > 0

    def test_profile_text_min_length_enforced(self):
        with pytest.raises(Exception):
            IdealCandidateProfile(
                tier=CandidateTier.MINIMUM,
                profile_text="Too short.",
                skills_demonstrated=[],
            )


class TestHyDEResult:
    def make_result(self) -> HyDEResult:
        long_text = "I am a software engineer with extensive Python and FastAPI experience. " * 4
        return HyDEResult(
            job_title="ML Engineer",
            profiles=[
                IdealCandidateProfile(tier=CandidateTier.MINIMUM, profile_text=long_text, skills_demonstrated=["Python"]),
                IdealCandidateProfile(tier=CandidateTier.STRONG, profile_text=long_text, skills_demonstrated=["Python", "Docker"]),
                IdealCandidateProfile(tier=CandidateTier.EXCEPTIONAL, profile_text=long_text, skills_demonstrated=["Python", "Docker", "Kubernetes"]),
            ],
        )

    def test_get_by_tier(self):
        result = self.make_result()
        assert result.get(CandidateTier.MINIMUM).tier == CandidateTier.MINIMUM
        assert result.strong.tier == CandidateTier.STRONG
        assert result.exceptional.tier == CandidateTier.EXCEPTIONAL

    def test_all_profile_texts_length(self):
        result = self.make_result()
        texts = result.all_profile_texts
        assert len(texts) == 3
        assert all(isinstance(t, str) for t in texts)

    def test_exactly_three_profiles_enforced(self):
        long_text = "I am a software engineer with extensive Python and FastAPI experience. " * 4
        with pytest.raises(Exception):
            HyDEResult(
                job_title="ML Engineer",
                profiles=[
                    IdealCandidateProfile(tier=CandidateTier.MINIMUM, profile_text=long_text, skills_demonstrated=[]),
                    IdealCandidateProfile(tier=CandidateTier.STRONG, profile_text=long_text, skills_demonstrated=[]),
                    # missing EXCEPTIONAL
                ],
            )

    def test_missing_tier_raises_key_error(self):
        long_text = "I am a software engineer with extensive Python and FastAPI experience. " * 4
        # Duplicate MINIMUM to force a missing STRONG
        result = HyDEResult(
            job_title="Test",
            profiles=[
                IdealCandidateProfile(tier=CandidateTier.MINIMUM, profile_text=long_text, skills_demonstrated=[]),
                IdealCandidateProfile(tier=CandidateTier.MINIMUM, profile_text=long_text, skills_demonstrated=[]),
                IdealCandidateProfile(tier=CandidateTier.EXCEPTIONAL, profile_text=long_text, skills_demonstrated=[]),
            ],
        )
        with pytest.raises(KeyError):
            result.get(CandidateTier.STRONG)





class TestFallbackGenerator:
    def test_produces_three_profiles(self):
        profile = make_profile(required=["Python", "Fastapi", "Docker"])
        result = _run_fallback(profile)
        assert len(result.profiles) == 3

    def test_all_tiers_present(self):
        profile = make_profile(required=["Python", "Fastapi"])
        result = _run_fallback(profile)
        tiers = {p.tier for p in result.profiles}
        assert tiers == {CandidateTier.MINIMUM, CandidateTier.STRONG, CandidateTier.EXCEPTIONAL}

    def test_minimum_has_only_required_skills(self):
        profile = make_profile(required=["Python", "Fastapi"], preferred=["Docker", "Kubernetes"])
        result = _run_fallback(profile)
        min_skills = set(result.minimum.skills_demonstrated)
        assert "Python" in min_skills
        assert "Docker" not in min_skills   # preferred — not in minimum

    def test_strong_includes_preferred_skills(self):
        profile = make_profile(required=["Python"], preferred=["Docker", "Kubernetes"])
        result = _run_fallback(profile)
        strong_skills = set(s.lower() for s in result.strong.skills_demonstrated)
        assert "docker" in strong_skills

    def test_exceptional_includes_all_skills(self):
        profile = make_profile(required=["Python", "Fastapi"], preferred=["Docker"])
        result = _run_fallback(profile)
        exc_skills = set(s.lower() for s in result.exceptional.skills_demonstrated)
        assert "python" in exc_skills
        assert "docker" in exc_skills

    def test_experience_years_increase_by_tier(self):
        profile = make_profile(years=3)
        result = _run_fallback(profile)
        assert result.minimum.experience_years <= result.strong.experience_years
        assert result.strong.experience_years <= result.exceptional.experience_years

    def test_profile_text_is_non_empty(self):
        profile = make_profile(required=["Python", "Fastapi", "Docker"])
        result = _run_fallback(profile)
        for p in result.profiles:
            assert len(p.profile_text.strip()) > 50

    def test_profile_text_contains_role(self):
        profile = make_profile(title="Data Scientist", required=["Python", "Machine Learning"])
        result = _run_fallback(profile)
        for p in result.profiles:
            assert "Data Scientist" in p.profile_text

    def test_differentiator_set(self):
        profile = make_profile()
        result = _run_fallback(profile)
        for p in result.profiles:
            assert len(p.differentiator) > 0

    def test_job_title_propagated(self):
        profile = make_profile(title="Senior Backend Engineer", required=["Python"])
        result = _run_fallback(profile)
        assert result.job_title == "Senior Backend Engineer"

    def test_domain_propagated(self):
        profile = make_profile(domain="FinTech", required=["Python"])
        result = _run_fallback(profile)
        assert result.domain == "FinTech"

    def test_senior_seniority_higher_years(self):
        senior = make_profile(seniority=SeniorityLevel.SENIOR, required=["Python"], years=None)
        junior = make_profile(seniority=SeniorityLevel.JUNIOR, required=["Python"], years=None)
        r_senior = _run_fallback(senior)
        r_junior = _run_fallback(junior)
        assert r_senior.minimum.experience_years >= r_junior.minimum.experience_years

    def test_empty_skills_still_produces_result(self):
        profile = make_profile(required=[], preferred=[])
        result = _run_fallback(profile)
        assert len(result.profiles) == 3

    def test_output_serialisable(self):
        profile = make_profile(required=["Python", "Fastapi"])
        result = _run_fallback(profile)
        data = result.model_dump()
        assert isinstance(data, dict)
        assert "profiles" in data
        assert len(data["profiles"]) == 3




class TestGenerateHyDEProfiles:
    def test_returns_hyde_result(self):
        profile = sample_profile()
        result = generate_hyde_profiles(profile, force_fallback=True)
        assert isinstance(result, HyDEResult)

    def test_end_to_end_from_sample_jd(self):
        profile = sample_profile()
        result = generate_hyde_profiles(profile, force_fallback=True)
        assert result.minimum.experience_years is not None
        assert result.exceptional.experience_years >= result.minimum.experience_years

    def test_all_profile_texts_unique(self):
        profile = sample_profile()
        result = generate_hyde_profiles(profile, force_fallback=True)
        texts = result.all_profile_texts
        # All three texts should be different
        assert len(set(texts)) == 3

    def test_pipeline_layer1_layer2_layer3(self):
        """Full chain: ingest → analyse_jd → generate_hyde_profiles."""
        from ai_hiring_ranker.ingestion.loader import ingest
        from ai_hiring_ranker.jd_intelligence.agent import analyse_jd
        from ai_hiring_ranker.hyde.generator import generate_hyde_profiles
        from pathlib import Path

        candidates_dir = ROOT / "data" / "sample" / "candidates"
        ingest_result = ingest(jd_path=SAMPLE_JD_PATH, candidates_dir=candidates_dir)
        hiring_profile = analyse_jd(ingest_result.jd, force_fallback=True)
        hyde_result = generate_hyde_profiles(hiring_profile, force_fallback=True)

        assert hyde_result.job_title != ""
        assert len(hyde_result.profiles) == 3
        assert all(p.word_count() > 20 for p in hyde_result.profiles)
