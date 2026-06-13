"""
Resume Claim Verification Agent — Layer 5.

Orchestrates claim verification for one candidate using all available
evidence sources, in priority order:

  1. Rule-based verification from resume text (always available)
  2. Project / achievement evidence from Layer 4 structured data
  3. GitHub API (if github_url available)
  4. Kaggle profile (if kaggle_url available)
  5. Portfolio links (if present)
  6. Optional LLM review for borderline claims (if API key available)

Public API:
  verify_candidate(profile, candidate_input, force_fallback=False)
    → VerificationReport

  verify_all_candidates(profiles, candidate_inputs, force_fallback=False)
    → list[VerificationReport]
"""

from __future__ import annotations

import logging
from typing import Optional

from ..candidate_extraction.schemas import CandidateProfile
from ..config import get_verification_config
from ..ingestion.schemas import CandidateInput
from .github_verifier import verify_skills_via_github
from .kaggle_verifier import verify_skills_via_kaggle
from .llm_verifier import apply_llm_adjustments
from .portfolio_verifier import verify_skills_via_portfolio
from .project_verifier import verify_skills_via_projects
from .rule_verifier import verify_candidate_rules
from .schemas import EvidenceItem, VerificationReport
from .utils import merge_external_evidence

logger = logging.getLogger(__name__)

_GITHUB_CONFIDENCE_BOOST = 0.20
_PROJECT_CONFIDENCE_BOOST = 0.15
_KAGGLE_CONFIDENCE_BOOST = 0.15
_PORTFOLIO_CONFIDENCE_BOOST = 0.10


def _merge_github_evidence(
    report: VerificationReport,
    github_evidence: list[EvidenceItem],
    candidate_id: str,
) -> VerificationReport:
    """Backward-compatible wrapper around the shared evidence merger."""
    del candidate_id
    merged = merge_external_evidence(
        report,
        github_evidence,
        confidence_boost=_GITHUB_CONFIDENCE_BOOST,
        config=get_verification_config(),
    )
    return merged.model_copy(update={"github_checked": True})


def verify_candidate(
    profile: CandidateProfile,
    candidate_input: Optional[CandidateInput] = None,
    *,
    force_fallback: bool = False,
) -> VerificationReport:
    """
    Run full claim verification for one candidate.

    Steps:
      1. Always run rule-based verification first (fast, no network).
      2. Merge project / achievement evidence from Layer 4.
      3. If external links are available and force_fallback=False,
         enrich with GitHub, Kaggle, and portfolio evidence.
      4. Optionally run LLM review for borderline claims.

    Args:
        profile:          CandidateProfile from Layer 4.
        candidate_input:  CandidateInput from Layer 1 (provides portfolio_links).
        force_fallback:   Skip external checks and LLM review (offline / testing).

    Returns:
        VerificationReport with fully populated VerifiedClaims.
    """
    config = get_verification_config()
    report = verify_candidate_rules(profile, config=config)
    logger.info(
        "[%s] Rule-based verification: %d claims, proof_strength=%.2f",
        profile.candidate_id,
        report.total_claims,
        report.proof_strength,
    )

    project_evidence = verify_skills_via_projects(profile)
    if project_evidence:
        report = merge_external_evidence(
            report,
            project_evidence,
            confidence_boost=_PROJECT_CONFIDENCE_BOOST,
            config=config,
        )
        logger.info(
            "[%s] Project evidence merged: %d items",
            profile.candidate_id,
            len(project_evidence),
        )

    github_url: Optional[str] = None
    kaggle_url: Optional[str] = None
    portfolio_urls: list[str] = []
    if candidate_input is not None:
        github_url = candidate_input.portfolio_links.github
        kaggle_url = candidate_input.portfolio_links.kaggle
        portfolio_urls = list(candidate_input.portfolio_links.portfolio)

    if force_fallback:
        logger.info(
            "[%s] Skipping external verification (force_fallback=True)",
            profile.candidate_id,
        )
        return apply_llm_adjustments(report, force_fallback=True)

    error_notes = list(report.error_notes)
    github_checked = report.github_checked

    if github_url:
        logger.info("[%s] Checking GitHub: %s", profile.candidate_id, github_url)
        github_checked = True
        try:
            gh_evidence, gh_errors = verify_skills_via_github(
                github_url=github_url,
                skills=profile.skill_names,
                candidate_id=profile.candidate_id,
                config=config,
            )
            error_notes.extend(gh_errors)
            if gh_evidence:
                report = merge_external_evidence(
                    report,
                    gh_evidence,
                    confidence_boost=_GITHUB_CONFIDENCE_BOOST,
                    config=config,
                )
                logger.info(
                    "[%s] GitHub merge complete: proof_strength=%.2f",
                    profile.candidate_id,
                    report.proof_strength,
                )
        except Exception as exc:
            error_notes.append(f"GitHub verification failed: {exc}")
            logger.warning("[%s] GitHub error: %s", profile.candidate_id, exc)

    if kaggle_url:
        logger.info("[%s] Checking Kaggle: %s", profile.candidate_id, kaggle_url)
        try:
            kg_evidence, kg_errors = verify_skills_via_kaggle(
                kaggle_url=kaggle_url,
                skills=profile.skill_names,
                candidate_id=profile.candidate_id,
            )
            error_notes.extend(kg_errors)
            if kg_evidence:
                report = merge_external_evidence(
                    report,
                    kg_evidence,
                    confidence_boost=_KAGGLE_CONFIDENCE_BOOST,
                    config=config,
                )
        except Exception as exc:
            error_notes.append(f"Kaggle verification failed: {exc}")
            logger.warning("[%s] Kaggle error: %s", profile.candidate_id, exc)

    if portfolio_urls:
        logger.info(
            "[%s] Checking %d portfolio links",
            profile.candidate_id,
            len(portfolio_urls),
        )
        try:
            pf_evidence, pf_errors = verify_skills_via_portfolio(
                portfolio_urls=portfolio_urls,
                skills=profile.skill_names,
                candidate_id=profile.candidate_id,
            )
            error_notes.extend(pf_errors)
            if pf_evidence:
                report = merge_external_evidence(
                    report,
                    pf_evidence,
                    confidence_boost=_PORTFOLIO_CONFIDENCE_BOOST,
                    config=config,
                )
        except Exception as exc:
            error_notes.append(f"Portfolio verification failed: {exc}")
            logger.warning("[%s] Portfolio error: %s", profile.candidate_id, exc)

    report = report.model_copy(
        update={
            "github_url": github_url,
            "kaggle_url": kaggle_url,
            "github_checked": github_checked,
            "error_notes": error_notes,
        }
    )

    report = apply_llm_adjustments(report, force_fallback=False)
    return report


def verify_all_candidates(
    profiles: list[CandidateProfile],
    candidate_inputs: Optional[list[CandidateInput]] = None,
    *,
    force_fallback: bool = False,
) -> list[VerificationReport]:
    """
    Verify all candidates in a batch.

    Failures for individual candidates are logged and a minimal report
    is returned rather than crashing the batch.
    """
    inputs_map: dict[str, CandidateInput] = {}
    if candidate_inputs:
        for candidate_input in candidate_inputs:
            inputs_map[candidate_input.candidate_id] = candidate_input

    reports: list[VerificationReport] = []
    for profile in profiles:
        candidate_input = inputs_map.get(profile.candidate_id)
        try:
            report = verify_candidate(
                profile,
                candidate_input,
                force_fallback=force_fallback,
            )
        except Exception as exc:
            logger.error(
                "Verification failed for %s: %s — returning minimal report.",
                profile.candidate_id,
                exc,
            )
            report = VerificationReport(
                candidate_id=profile.candidate_id,
                candidate_name=profile.name,
                error_notes=[f"Verification crashed: {exc}"],
            )
        reports.append(report)

    return reports
