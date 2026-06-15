"""
Evidence Ledger builder — Layer 6.

Converts the output of Layer 5 (VerificationReport) into an immutable,
auditable CandidateLedger / RunLedger. Also handles persistence:
reading and writing JSON files to outputs/.

Public API
----------
build_candidate_ledger(report, candidate_name, run_id)
    → CandidateLedger

build_run_ledger(reports, job_title, run_id)
    → RunLedger

save_run_ledger(run_ledger, output_dir)
    → Path  (path to the written JSON file)

load_run_ledger(path)
    → RunLedger

load_candidate_ledger(path)
    → CandidateLedger
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..claim_verification.schemas import EvidenceSource, VerificationReport, VerifiedClaim
from ..ingestion.schemas import VerificationStatus
from .schemas import (
    CandidateLedger,
    ClaimSource,
    LedgerEntry,
    RunLedger,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# claim_id generation
# ---------------------------------------------------------------------------


def make_claim_id(candidate_id: str, skill: str, claim_text: str) -> str:
    """
    Generate a stable, deterministic claim ID.

    Uses SHA-256 of (candidate_id + skill + claim_text) so the same claim
    always gets the same ID across different runs, enabling deduplication
    and diff-based auditing.
    """
    raw = f"{candidate_id}|{skill.strip().lower()}|{claim_text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# EvidenceSource → ClaimSource mapping
# ---------------------------------------------------------------------------


def _map_source(evidence_source: EvidenceSource) -> ClaimSource:
    """Map a Layer 5 EvidenceSource to a Layer 6 ClaimSource."""
    _MAP = {
        EvidenceSource.RESUME:        ClaimSource.RESUME,
        EvidenceSource.GITHUB_REPO:   ClaimSource.GITHUB,
        EvidenceSource.GITHUB_COMMIT: ClaimSource.GITHUB,
        EvidenceSource.GITHUB_FILE:   ClaimSource.GITHUB,
        EvidenceSource.KAGGLE:        ClaimSource.KAGGLE,
        EvidenceSource.PORTFOLIO:     ClaimSource.PORTFOLIO,
        EvidenceSource.INFERRED:      ClaimSource.INFERRED,
    }
    return _MAP.get(evidence_source, ClaimSource.RESUME)


def _best_source(claim: VerifiedClaim) -> ClaimSource:
    """
    Determine the primary source for a VerifiedClaim by picking the
    highest-confidence evidence item (non-resume sources preferred).
    """
    if not claim.evidence:
        return ClaimSource.RESUME

    # Prefer external sources over plain resume text
    priority_order = [
        EvidenceSource.GITHUB_REPO,
        EvidenceSource.GITHUB_COMMIT,
        EvidenceSource.GITHUB_FILE,
        EvidenceSource.KAGGLE,
        EvidenceSource.PORTFOLIO,
        EvidenceSource.INFERRED,
        EvidenceSource.RESUME,
    ]
    for source in priority_order:
        match = next((e for e in claim.evidence if e.source == source), None)
        if match:
            return _map_source(source)

    return _map_source(claim.evidence[0].source)


# ---------------------------------------------------------------------------
# VerifiedClaim → LedgerEntry
# ---------------------------------------------------------------------------


def _claim_to_entry(claim: VerifiedClaim) -> LedgerEntry:
    """Convert one VerifiedClaim (Layer 5) into a LedgerEntry (Layer 6)."""

    # Pick the best evidence item for URL and snippet
    best_evidence_item = None
    if claim.evidence:
        best_evidence_item = max(claim.evidence, key=lambda e: e.relevance_score)

    evidence_url     = best_evidence_item.url     if best_evidence_item else None
    evidence_snippet = best_evidence_item.snippet if best_evidence_item else ""
    recency_years    = best_evidence_item.recency_years if best_evidence_item else None

    # Build notes: collect any extra context worth preserving
    notes_parts: list[str] = []
    if recency_years is not None and recency_years > 3:
        notes_parts.append(f"Evidence is {recency_years:.1f} years old — recency penalty may apply.")
    if len(claim.evidence) > 1:
        notes_parts.append(f"{len(claim.evidence)} evidence items collected.")
    if claim.status == VerificationStatus.PENDING:
        notes_parts.append("Verification was not completed (no external links or API unavailable).")

    return LedgerEntry(
        claim_id=make_claim_id(claim.candidate_id, claim.skill, claim.claim_text),
        candidate_id=claim.candidate_id,
        skill=claim.skill,
        claim_text=claim.claim_text,
        source=_best_source(claim),
        verification_status=claim.status,
        confidence=claim.confidence,
        evidence_url=evidence_url,
        evidence_snippet=evidence_snippet,
        recency_years=recency_years,
        reasoning=claim.reasoning,
        notes=" ".join(notes_parts) if notes_parts else None,
        verified_at=claim.verified_at,
        ledger_created_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_candidate_ledger(
    report: VerificationReport,
    *,
    candidate_name: str = "",
    run_id: str = "",
) -> CandidateLedger:
    """
    Build a CandidateLedger from a Layer 5 VerificationReport.

    Args:
        report:         VerificationReport from verify_candidate().
        candidate_name: Display name (taken from report if empty).
        run_id:         Pipeline run identifier for traceability.

    Returns:
        CandidateLedger with one LedgerEntry per VerifiedClaim.
    """
    name = candidate_name or report.candidate_name or report.candidate_id
    entries: list[LedgerEntry] = []

    for claim in report.claims:
        try:
            entries.append(_claim_to_entry(claim))
        except Exception as exc:
            logger.warning(
                "[%s] Failed to convert claim '%s' to ledger entry: %s — skipping.",
                report.candidate_id,
                claim.skill,
                exc,
            )

    ledger = CandidateLedger(
        candidate_id=report.candidate_id,
        candidate_name=name,
        entries=entries,
        run_id=run_id,
        created_at=datetime.utcnow(),
    )

    logger.info(
        "[%s] Ledger built: %d entries, proof_strength=%.2f",
        ledger.candidate_id,
        ledger.total_claims,
        ledger.proof_strength,
    )
    return ledger


def build_run_ledger(
    reports: list[VerificationReport],
    *,
    job_title: str = "",
    run_id: Optional[str] = None,
) -> RunLedger:
    """
    Build a RunLedger from a batch of Layer 5 VerificationReports.

    Args:
        reports:   List of VerificationReport objects (one per candidate).
        job_title: Job title string from the JD (for audit trail labelling).
        run_id:    Explicit run ID; auto-generated UUID4 if None.

    Returns:
        RunLedger containing one CandidateLedger per candidate.
    """
    resolved_run_id = run_id or str(uuid.uuid4())[:8]

    candidate_ledgers: list[CandidateLedger] = []
    for report in reports:
        try:
            ledger = build_candidate_ledger(
                report,
                run_id=resolved_run_id,
            )
            candidate_ledgers.append(ledger)
        except Exception as exc:
            logger.error(
                "Failed to build ledger for candidate '%s': %s — inserting empty ledger.",
                report.candidate_id,
                exc,
            )
            candidate_ledgers.append(
                CandidateLedger(
                    candidate_id=report.candidate_id,
                    candidate_name=report.candidate_name,
                    run_id=resolved_run_id,
                )
            )

    run_ledger = RunLedger(
        run_id=resolved_run_id,
        job_title=job_title,
        candidates=candidate_ledgers,
        created_at=datetime.utcnow(),
    )

    logger.info(
        "RunLedger[%s] built: %d candidates, %d total claims.",
        resolved_run_id,
        run_ledger.candidate_count,
        run_ledger.total_claims,
    )
    return run_ledger


# ---------------------------------------------------------------------------
# Persistence — save / load
# ---------------------------------------------------------------------------


def save_run_ledger(
    run_ledger: RunLedger,
    output_dir: Path | str,
    *,
    pretty: bool = True,
) -> Path:
    """
    Persist a RunLedger to disk as JSON.

    File is written to:
        <output_dir>/<run_id>_ledger.json

    Args:
        run_ledger:  The RunLedger to write.
        output_dir:  Directory to write into (created if missing).
        pretty:      Indent the JSON for human readability (default True).

    Returns:
        Path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = f"{run_ledger.run_id}_ledger.json"
    path = out / filename

    payload = run_ledger.to_export_dict()
    indent = 2 if pretty else None

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False)

    logger.info("RunLedger[%s] saved → %s", run_ledger.run_id, path)
    return path


def save_candidate_ledger(
    ledger: CandidateLedger,
    output_dir: Path | str,
    *,
    pretty: bool = True,
) -> Path:
    """
    Persist a single CandidateLedger to disk as JSON.

    File is written to:
        <output_dir>/<candidate_id>_ledger.json

    Useful for incremental/streaming runs where you want to save
    each candidate's ledger as soon as it's built.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = f"{ledger.candidate_id}_ledger.json"
    path = out / filename

    payload = ledger.to_export_dict()
    indent = 2 if pretty else None

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False)

    logger.info("CandidateLedger[%s] saved → %s", ledger.candidate_id, path)
    return path


def load_run_ledger(path: Path | str) -> RunLedger:
    """
    Load a RunLedger from a JSON file previously written by save_run_ledger().

    The file is expected to match the export format produced by
    RunLedger.to_export_dict(). Raises FileNotFoundError if path is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ledger file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Re-hydrate: convert export dicts back to typed models
    candidates: list[CandidateLedger] = []
    for cand_dict in raw.get("candidates", []):
        entries: list[LedgerEntry] = []
        for entry_dict in cand_dict.get("claims", []):
            entries.append(
                LedgerEntry(
                    claim_id=entry_dict["claim_id"],
                    candidate_id=cand_dict["candidate_id"],
                    skill=entry_dict["skill"],
                    claim_text=entry_dict.get("claim_text", ""),
                    source=ClaimSource(entry_dict.get("source", "resume")),
                    verification_status=VerificationStatus(entry_dict["verification_status"]),
                    confidence=entry_dict.get("confidence", 0.0),
                    evidence_url=entry_dict.get("evidence_url"),
                    recency_years=entry_dict.get("recency_years"),
                    notes=entry_dict.get("notes"),
                )
            )
        candidates.append(
            CandidateLedger(
                candidate_id=cand_dict["candidate_id"],
                entries=entries,
            )
        )

    return RunLedger(
        run_id=raw.get("run_id", "unknown"),
        job_title=raw.get("job_title", ""),
        candidates=candidates,
        created_at=datetime.fromisoformat(raw["created_at"])
        if "created_at" in raw
        else datetime.utcnow(),
    )


def load_candidate_ledger(path: Path | str) -> CandidateLedger:
    """
    Load a CandidateLedger from a JSON file previously written by
    save_candidate_ledger().
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ledger file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    candidate_id = raw["candidate_id"]
    entries: list[LedgerEntry] = []
    for entry_dict in raw.get("claims", []):
        entries.append(
            LedgerEntry(
                claim_id=entry_dict["claim_id"],
                candidate_id=candidate_id,
                skill=entry_dict["skill"],
                claim_text=entry_dict.get("claim_text", ""),
                source=ClaimSource(entry_dict.get("source", "resume")),
                verification_status=VerificationStatus(entry_dict["verification_status"]),
                confidence=entry_dict.get("confidence", 0.0),
                evidence_url=entry_dict.get("evidence_url"),
                recency_years=entry_dict.get("recency_years"),
                notes=entry_dict.get("notes"),
            )
        )

    return CandidateLedger(candidate_id=candidate_id, entries=entries)
