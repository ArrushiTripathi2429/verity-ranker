"""
LLM claim verifier — optional enrichment for borderline claims.

Uses OpenAI structured output to review ambiguous claims with resume and
external evidence context. Falls back silently when no API key is present.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..config import get_llm_config
from ..ingestion.schemas import VerificationStatus
from ..llm_provider import structured_completion
from .schemas import VerificationReport, VerifiedClaim
from .utils import status_from_confidence

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[4] / "prompts" / "claim_verification.md"

_BORDERLINE_STATUSES = {
    VerificationStatus.WEAK,
    VerificationStatus.INFERRED,
    VerificationStatus.UNSUPPORTED,
}


class LLMClaimAdjustment(BaseModel):
    skill: str
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class LLMVerificationResult(BaseModel):
    adjustments: list[LLMClaimAdjustment] = Field(default_factory=list)


def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a technical claim verifier. Review candidate skill claims and "
        "return JSON adjustments for borderline claims only."
    )


def _build_user_prompt(report: VerificationReport) -> str:
    claims_payload = []
    for claim in report.claims:
        if claim.status not in _BORDERLINE_STATUSES:
            continue
        claims_payload.append(
            {
                "skill": claim.skill,
                "claim_text": claim.claim_text,
                "current_status": claim.status.value,
                "current_confidence": claim.confidence,
                "evidence": [
                    {
                        "source": item.source.value,
                        "snippet": item.snippet,
                        "url": item.url,
                        "relevance_score": item.relevance_score,
                    }
                    for item in claim.evidence[:5]
                ],
            }
        )

    return (
        "Review these borderline skill claims and decide whether the current "
        "status/confidence should be adjusted.\n\n"
        f"{json.dumps({'candidate_id': report.candidate_id, 'claims': claims_payload}, indent=2)}"
    )


def apply_llm_adjustments(
    report: VerificationReport,
    *,
    force_fallback: bool = False,
) -> VerificationReport:
    """Optionally refine borderline claims using the configured LLM."""
    if force_fallback:
        return report

    borderline = [c for c in report.claims if c.status in _BORDERLINE_STATUSES]
    if not borderline:
        return report

    cfg = get_llm_config()
    if not cfg.api_key:
        logger.info("[%s] Skipping LLM verification (no API key)", report.candidate_id)
        return report

    try:
        result = structured_completion(
            _load_system_prompt(),
            _build_user_prompt(report),
            LLMVerificationResult,
        )
    except Exception as exc:
        logger.warning(
            "[%s] LLM verification failed, keeping rule-based result: %s",
            report.candidate_id,
            exc,
        )
        report.error_notes.append(f"LLM verification skipped: {exc}")
        return report

    if not result.adjustments:
        return report

    adjustments = {adj.skill.strip().title(): adj for adj in result.adjustments}
    updated_claims: list[VerifiedClaim] = []

    for claim in report.claims:
        adj = adjustments.get(claim.skill)
        if adj is None:
            updated_claims.append(claim)
            continue

        updated_claims.append(
            claim.model_copy(
                update={
                    "status": adj.status,
                    "confidence": round(adj.confidence, 3),
                    "reasoning": f"{claim.reasoning}; LLM review: {adj.reasoning}",
                }
            )
        )

    return report.model_copy(update={"claims": updated_claims})
