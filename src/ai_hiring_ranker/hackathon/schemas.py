"""Schemas for hackathon precompute cache and submission output."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PrecomputeManifest(BaseModel):
    """Metadata written by precompute.py."""

    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    jd_path: str = ""
    candidates_path: str = ""
    candidate_count: int = 0
    force_fallback: bool = True
    used_llm: bool = False
    job_title: str = ""
    cache_file: str = "candidate_features.jsonl"
    jd_cache_file: str = "jd_profile.json"


class SubmissionRow(BaseModel):
    candidate_id: str
    rank: int = Field(ge=1, le=100)
    score: float = Field(ge=0.0, le=100.0)
    reasoning: str
