"""
Layer 6 — Evidence Ledger.

Converts Layer 5 VerificationReports into immutable, auditable ledgers
that every downstream layer reads instead of re-running verification.

Public API
----------
from ai_hiring_ranker.evidence import (
    # Builders
    build_candidate_ledger,
    build_run_ledger,

    # Persistence
    save_run_ledger,
    save_candidate_ledger,
    load_run_ledger,
    load_candidate_ledger,

    # Schema types
    LedgerEntry,
    CandidateLedger,
    RunLedger,
    ClaimSource,
)
"""

from .ledger import (
    build_candidate_ledger,
    build_run_ledger,
    load_candidate_ledger,
    load_run_ledger,
    make_claim_id,
    save_candidate_ledger,
    save_run_ledger,
)
from .schemas import (
    CandidateLedger,
    ClaimSource,
    LedgerEntry,
    RunLedger,
)

__all__ = [
    # Builders
    "build_candidate_ledger",
    "build_run_ledger",
    # Persistence
    "save_run_ledger",
    "save_candidate_ledger",
    "load_run_ledger",
    "load_candidate_ledger",
    # Utilities
    "make_claim_id",
    # Schema types
    "LedgerEntry",
    "CandidateLedger",
    "RunLedger",
    "ClaimSource",
]
