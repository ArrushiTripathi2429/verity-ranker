"""
Tests for Layer 5 — Resume Claim Verification Agent.

All tests run in force_fallback=True (no GitHub API calls).
GitHub merging logic is tested with injected mock evidence.
Covers: schemas, rule verifier, agent orchestration, full pipeline L1→L5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_hiring_ranker.ingestion.loader import ingest
from ai_hiring_ranker.ingestion.schemas import CandidateInput, VerificationStatus
from ai_hiring_ranker.candidate_extraction.extractor import (
    extract_candidate_profile,
    extract_all_candidates,
)
from ai_hiring_ranker.candidate_extraction.schemas import (
    CandidateProfile,
    ProjectEntry,
    SkillClaim,
    SkillConfidence,
)
from ai_hiring_ranker.claim_verification.schemas import (
    EvidenceItem,
    EvidenceSource,
    VerificationReport,
    VerifiedClaim,
)
from ai_hiring_ranker.claim_verification.rule_verifier import (
    verify_candidate_rules,
    _compute_status_and_confidence,
    _is_negated,
    _has_production_evidence,
    _has_metric,
    _snippets_are_recent,
)
from ai_hiring_ranker.claim_verification.agent import (
    verify_candidate,
    verify_all_candidates,
    _merge_github_evidence,
)
from ai_hiring_ranker.claim_verification.project_verifier import verify_skills_via_projects
from ai_hiring_ranker.claim_verification.utils import (
    apply_recency_penalty,
    evidence_matches_skill,
    status_from_confidence,
)
from ai_hiring_ranker.config import get_verification_config

SAMPLE_JD         = ROOT / "data" / "sample" / "jd.txt"
SAMPLE_CANDIDATES = ROOT / "data" / "sample" / "candidates"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidate_input(text: str, cid: str = "T001", github: str = None) -> CandidateInput:
    from ai_hiring_ranker.ingestion.schemas import PortfolioLinks
    pl = PortfolioLinks(github=github)
    return CandidateInput(candidate_id=cid, raw_text=text, portfolio_links=pl)


def make_profile(skills: list[tuple[str, SkillConfidence, list[str]]], cid: str = "T001") -> CandidateProfile:
    claims = [
        SkillClaim(skill=s, confidence=c, evidence_snippets=e)
        for s, c, e in skills
    ]
    return CandidateProfile(candidate_id=cid, skills=claims)


def load_and_extract(candidate_id: str) -> tuple[CandidateInput, CandidateProfile]:
    result = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
    ci = next(c for c in result.candidates if c.candidate_id == candidate_id)
    profile = extract_candidate_profile(ci, force_fallback=True)
    return ci, profile


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestVerifiedClaim:
    def test_is_verified_true(self):
        claim = VerifiedClaim(
            candidate_id="X", skill="Python",
            status=VerificationStatus.VERIFIED, confidence=0.9,
        )
        assert claim.is_verified is True

    def test_is_verified_false(self):
        claim = VerifiedClaim(
            candidate_id="X", skill="Python",
            status=VerificationStatus.WEAK, confidence=0.5,
        )
        assert claim.is_verified is False

    def test_is_supported_weak(self):
        claim = VerifiedClaim(
            candidate_id="X", skill="Python",
            status=VerificationStatus.WEAK, confidence=0.5,
        )
        assert claim.is_supported is True

    def test_is_supported_unsupported(self):
        claim = VerifiedClaim(
            candidate_id="X", skill="Python",
            status=VerificationStatus.UNSUPPORTED, confidence=0.0,
        )
        assert claim.is_supported is False

    def test_best_evidence_url_none_when_no_evidence(self):
        claim = VerifiedClaim(candidate_id="X", skill="Python")
        assert claim.best_evidence_url is None

    def test_best_evidence_url_returns_first(self):
        claim = VerifiedClaim(
            candidate_id="X", skill="Python",
            evidence=[
                EvidenceItem(source=EvidenceSource.GITHUB_REPO, url="https://github.com/u", snippet="x"),
            ],
        )
        assert claim.best_evidence_url == "https://github.com/u"

    def test_skill_normalised(self):
        claim = VerifiedClaim(candidate_id="X", skill="python")
        assert claim.skill == "Python"


class TestVerificationReport:
    def _make_report(self) -> VerificationReport:
        return VerificationReport(
            candidate_id="C001",
            claims=[
                VerifiedClaim(candidate_id="C001", skill="Python",  status=VerificationStatus.VERIFIED,    confidence=0.9),
                VerifiedClaim(candidate_id="C001", skill="Docker",  status=VerificationStatus.WEAK,        confidence=0.5),
                VerifiedClaim(candidate_id="C001", skill="Sql",     status=VerificationStatus.INFERRED,    confidence=0.3),
                VerifiedClaim(candidate_id="C001", skill="Cobol",   status=VerificationStatus.UNSUPPORTED, confidence=0.0),
            ],
        )

    def test_counts(self):
        r = self._make_report()
        assert r.verified_count   == 1
        assert r.weak_count       == 1
        assert r.inferred_count   == 1
        assert r.unsupported_count== 1
        assert r.total_claims     == 4

    def test_proof_strength_range(self):
        r = self._make_report()
        assert 0.0 <= r.proof_strength <= 1.0

    def test_proof_strength_all_verified(self):
        r = VerificationReport(
            candidate_id="X",
            claims=[
                VerifiedClaim(candidate_id="X", skill="Python", status=VerificationStatus.VERIFIED, confidence=0.9),
                VerifiedClaim(candidate_id="X", skill="Docker", status=VerificationStatus.VERIFIED, confidence=0.85),
            ],
        )
        assert r.proof_strength == 1.0

    def test_proof_strength_empty(self):
        r = VerificationReport(candidate_id="X")
        assert r.proof_strength == 0.0

    def test_verified_skill_names(self):
        r = self._make_report()
        assert "Python" in r.verified_skill_names
        assert "Docker" not in r.verified_skill_names

    def test_get_claim_found(self):
        r = self._make_report()
        claim = r.get_claim("python")  # case-insensitive normalisation
        assert claim is not None
        assert claim.skill == "Python"

    def test_get_claim_missing(self):
        r = self._make_report()
        assert r.get_claim("Rust") is None

    def test_summary_line_runs(self):
        r = self._make_report()
        line = r.summary_line()
        assert "C001" in line
        assert "proof_strength" in line


# ---------------------------------------------------------------------------
# Rule-verifier unit tests
# ---------------------------------------------------------------------------


class TestRuleVerifierHelpers:
    def test_is_negated_true(self):
        assert _is_negated(["I have no experience with Docker."]) is True

    def test_is_negated_false(self):
        assert _is_negated(["I deployed a Docker container to production."]) is False

    def test_is_negated_empty(self):
        assert _is_negated([]) is False

    def test_has_production_evidence_true(self):
        assert _has_production_evidence(["Deployed the service to production."]) is True

    def test_has_production_evidence_false(self):
        assert _has_production_evidence(["Built a notebook prototype."]) is False

    def test_has_metric_true(self):
        assert _has_metric(["Reduced latency by 40%."]) is True

    def test_has_metric_false(self):
        assert _has_metric(["Improved overall system performance."]) is False

    def test_snippets_are_recent_with_year(self):
        assert _snippets_are_recent(["Used Python in 2024 for production services."]) is True

    def test_snippets_are_recent_with_keyword(self):
        assert _snippets_are_recent(["Currently building ML pipelines."]) is True

    def test_snippets_are_recent_old(self):
        assert _snippets_are_recent(["Worked with Python in 2010."]) is False


class TestComputeStatusAndConfidence:
    def make_claim(
        self,
        snippets: list[str],
        confidence: SkillConfidence = SkillConfidence.EXPLICIT,
    ) -> SkillClaim:
        return SkillClaim(skill="Python", confidence=confidence, evidence_snippets=snippets)

    def test_negated_gives_unsupported(self):
        claim = self.make_claim(["I have no experience with Python."])
        status, conf, _ = _compute_status_and_confidence(claim, negated=True)
        assert status == VerificationStatus.UNSUPPORTED
        assert conf == 0.0

    def test_strong_evidence_verified(self):
        claim = self.make_claim([
            "Built production Python ML services deployed in 2024.",
            "Improved model accuracy by 15% using Python optimisations.",
        ])
        status, conf, _ = _compute_status_and_confidence(claim, negated=False)
        assert status == VerificationStatus.VERIFIED
        assert conf >= 0.75

    def test_weak_confidence_no_snippets_unsupported(self):
        claim = self.make_claim([], confidence=SkillConfidence.WEAK)
        status, conf, _ = _compute_status_and_confidence(claim, negated=False)
        assert status == VerificationStatus.UNSUPPORTED

    def test_one_snippet_inferred_or_weak(self):
        claim = self.make_claim(
            ["Used Python for data analysis."],
            confidence=SkillConfidence.INFERRED,
        )
        status, conf, _ = _compute_status_and_confidence(claim, negated=False)
        assert status in (VerificationStatus.WEAK, VerificationStatus.INFERRED)


class TestRuleVerifier:
    def test_c001_has_python_verified_or_weak(self):
        _, profile = load_and_extract("C001")
        report = verify_candidate_rules(profile)
        python_claim = report.get_claim("Python")
        assert python_claim is not None
        assert python_claim.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.WEAK,
            VerificationStatus.INFERRED,
        )

    def test_report_has_one_claim_per_skill(self):
        _, profile = load_and_extract("C001")
        report = verify_candidate_rules(profile)
        skill_names = [c.skill for c in report.claims]
        assert len(skill_names) == len(set(skill_names)), "Duplicate claims found"

    def test_all_statuses_valid_enum(self):
        _, profile = load_and_extract("C001")
        report = verify_candidate_rules(profile)
        valid = set(VerificationStatus)
        for claim in report.claims:
            assert claim.status in valid

    def test_proof_strength_gt_zero_for_strong_candidate(self):
        _, profile = load_and_extract("C001")
        report = verify_candidate_rules(profile)
        assert report.proof_strength > 0.0

    def test_c003_lower_proof_than_c001(self):
        # C003 is a junior with limited evidence
        _, p1 = load_and_extract("C001")
        _, p3 = load_and_extract("C003")
        r1 = verify_candidate_rules(p1)
        r3 = verify_candidate_rules(p3)
        assert r1.proof_strength >= r3.proof_strength

    def test_evidence_items_have_resume_source(self):
        _, profile = load_and_extract("C001")
        report = verify_candidate_rules(profile)
        for claim in report.claims:
            for ev in claim.evidence:
                assert ev.source == EvidenceSource.RESUME

    def test_github_not_checked_in_rule_mode(self):
        _, profile = load_and_extract("C001")
        report = verify_candidate_rules(profile)
        assert report.github_checked is False


# ---------------------------------------------------------------------------
# Agent orchestration tests
# ---------------------------------------------------------------------------


class TestVerifyCandidate:
    def test_returns_report(self):
        ci, profile = load_and_extract("C001")
        report = verify_candidate(profile, ci, force_fallback=True)
        assert isinstance(report, VerificationReport)

    def test_force_fallback_skips_github(self):
        ci, profile = load_and_extract("C001")
        report = verify_candidate(profile, ci, force_fallback=True)
        assert report.github_checked is False

    def test_no_candidate_input_still_works(self):
        _, profile = load_and_extract("C001")
        report = verify_candidate(profile, None, force_fallback=True)
        assert report.total_claims > 0

    def test_github_merge_boosts_confidence(self):
        # Inject mock GitHub evidence directly into _merge_github_evidence
        _, profile = load_and_extract("C001")
        base_report = verify_candidate_rules(profile)

        mock_gh_evidence = [
            EvidenceItem(
                source=EvidenceSource.GITHUB_REPO,
                url="https://github.com/testuser",
                snippet="python fastapi docker languages detected: Python",
                relevance_score=0.9,
                recency_years=0.5,
            )
        ]
        merged = _merge_github_evidence(base_report, mock_gh_evidence, "C001")
        # proof_strength should be >= base (never lower after positive evidence)
        assert merged.proof_strength >= base_report.proof_strength

    def test_github_merge_sets_github_checked(self):
        _, profile = load_and_extract("C001")
        base_report = verify_candidate_rules(profile)
        merged = _merge_github_evidence(base_report, [], "C001")
        assert merged.github_checked is True


class TestVerifyAllCandidates:
    def test_returns_one_report_per_profile(self):
        result   = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        profiles = extract_all_candidates(result.candidates, force_fallback=True)
        reports  = verify_all_candidates(profiles, result.candidates, force_fallback=True)
        assert len(reports) == len(profiles)

    def test_all_reports_are_verification_report(self):
        result   = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        profiles = extract_all_candidates(result.candidates, force_fallback=True)
        reports  = verify_all_candidates(profiles, force_fallback=True)
        assert all(isinstance(r, VerificationReport) for r in reports)

    def test_candidate_ids_preserved(self):
        result   = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        profiles = extract_all_candidates(result.candidates, force_fallback=True)
        reports  = verify_all_candidates(profiles, result.candidates, force_fallback=True)
        profile_ids = {p.candidate_id for p in profiles}
        report_ids  = {r.candidate_id for r in reports}
        assert profile_ids == report_ids


# ---------------------------------------------------------------------------
# End-to-end: Layer 1 → 2 → 3 → 4 → 5
# ---------------------------------------------------------------------------


class TestEndToEndL1toL5:
    def test_full_pipeline(self):
        from ai_hiring_ranker.jd_intelligence.agent import analyse_jd
        from ai_hiring_ranker.hyde.generator import generate_hyde_profiles

        ingest_result  = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        hiring_profile = analyse_jd(ingest_result.jd, force_fallback=True)
        hyde_result    = generate_hyde_profiles(hiring_profile, force_fallback=True)
        profiles       = extract_all_candidates(ingest_result.candidates, force_fallback=True)
        reports        = verify_all_candidates(profiles, ingest_result.candidates, force_fallback=True)

        assert len(reports) == ingest_result.candidate_count
        assert all(r.proof_strength >= 0.0 for r in reports)
        assert all(r.total_claims > 0 for r in reports)

    def test_proof_strength_ordering_makes_sense(self):
        """C001 (strong ML engineer) should have higher proof than C003 (junior)."""
        from ai_hiring_ranker.jd_intelligence.agent import analyse_jd

        ingest_result  = ingest(jd_path=SAMPLE_JD, candidates_dir=SAMPLE_CANDIDATES)
        profiles       = extract_all_candidates(ingest_result.candidates, force_fallback=True)
        reports        = verify_all_candidates(profiles, force_fallback=True)

        by_id = {r.candidate_id: r for r in reports}
        # C001 has more evidence than C003 — proof strength should be higher
        assert by_id["C001"].proof_strength >= by_id["C003"].proof_strength


class TestVerificationConfig:
    def test_loads_yaml_thresholds(self):
        cfg = get_verification_config()
        assert cfg.github.recency_cutoff_years == 3
        assert cfg.verification_labels.verified.min_confidence == 0.75
        assert cfg.verification_labels.weak.min_confidence == 0.40


class TestVerificationUtils:
    def test_status_from_confidence_verified(self):
        assert status_from_confidence(0.9) == VerificationStatus.VERIFIED

    def test_status_from_confidence_unsupported(self):
        assert status_from_confidence(0.05) == VerificationStatus.UNSUPPORTED

    def test_evidence_matches_skill_by_tag(self):
        item = EvidenceItem(
            source=EvidenceSource.GITHUB_REPO,
            skill="Python",
            snippet="languages detected",
        )
        assert evidence_matches_skill(item, "Python") is True
        assert evidence_matches_skill(item, "Docker") is False

    def test_apply_recency_penalty_downgrades_old_evidence(self):
        item = EvidenceItem(
            source=EvidenceSource.GITHUB_REPO,
            skill="Python",
            snippet="old repo",
            recency_years=6.0,
            relevance_score=0.8,
        )
        penalised = apply_recency_penalty(item, recency_cutoff_years=3)
        assert penalised.relevance_score < item.relevance_score


class TestProjectVerifier:
    def test_project_evidence_tags_skill(self):
        profile = make_profile(
            [
                ("Python", SkillConfidence.EXPLICIT, ["Built Python services."]),
            ]
        )
        profile.projects = [
            ProjectEntry(
                title="ML API",
                description="Built a FastAPI inference service in Python.",
                skills_used=["Python", "FastAPI"],
                is_production=True,
                has_metrics=False,
            )
        ]
        evidence = verify_skills_via_projects(profile)
        assert any(item.skill == "Python" for item in evidence)
