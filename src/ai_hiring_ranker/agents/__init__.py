"""
Layer 9 — Multi-Agent Evaluation.

Five specialist agents evaluate every shortlisted candidate across
independent dimensions, then a synthesis step combines them into
a single EvaluationResult with six rubric dimension scores.

Agents:
  1. JD Fit Agent          — skill / requirement coverage
  2. Technical Fit Agent   — depth, production evidence, verification
  3. Career Trajectory Agent — seniority, experience, growth
  4. Verification Agent    — evidence ledger proof strength
  (5. Final Synthesis       — runs inside the orchestrator)

Public API
----------
from ai_hiring_ranker.agents import (
    evaluate_all,        # batch entry point → BatchEvaluationResult
    evaluate_candidate,  # single candidate → EvaluationResult
    BatchEvaluationResult,
    EvaluationResult,
    AgentVerdict,
    DimensionScores,
    AgentRole,
)
"""

from .orchestrator import evaluate_all, evaluate_candidate
from .schemas import (
    AgentRole,
    AgentVerdict,
    BatchEvaluationResult,
    DimensionScores,
    EvaluationResult,
)

__all__ = [
    # Orchestrator
    "evaluate_all",
    "evaluate_candidate",
    # Schema types
    "BatchEvaluationResult",
    "EvaluationResult",
    "AgentVerdict",
    "DimensionScores",
    "AgentRole",
]
