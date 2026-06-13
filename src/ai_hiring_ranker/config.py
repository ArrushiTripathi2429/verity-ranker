"""
Central config loader for ai-hiring-ranker.

Reads configs/v2/models.yaml and other YAML configs.
Provides typed config objects used by all agents.
Falls back to safe defaults if a config file is missing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Optional YAML support — falls back to manual parsing if pyyaml not installed
try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

ROOT = Path(__file__).resolve().parents[3]   # repo root
CONFIGS_DIR = ROOT / "configs" / "v2"


# ---------------------------------------------------------------------------
# Typed config models
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536


class HyDEConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    num_profiles: int = 3
    temperature: float = 0.7


class GitHubVerificationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    check_commits: bool = True
    check_file_types: bool = True
    check_readme: bool = True
    check_tests: bool = True
    recency_cutoff_years: int = 3


class VerificationLabelThreshold(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str = ""
    min_confidence: float = 0.0


class VerificationLabelsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verified: VerificationLabelThreshold = Field(
        default_factory=lambda: VerificationLabelThreshold(
            description="Direct code/commit evidence found; recent and relevant.",
            min_confidence=0.75,
        )
    )
    weak: VerificationLabelThreshold = Field(
        default_factory=lambda: VerificationLabelThreshold(
            description="Evidence exists but is indirect, old, or low volume.",
            min_confidence=0.40,
        )
    )
    inferred: VerificationLabelThreshold = Field(
        default_factory=lambda: VerificationLabelThreshold(
            description="Skill inferred from adjacent evidence.",
            min_confidence=0.20,
        )
    )
    unsupported: VerificationLabelThreshold = Field(
        default_factory=lambda: VerificationLabelThreshold(
            description="No evidence found for the claim.",
            min_confidence=0.0,
        )
    )


class VerificationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    github: GitHubVerificationConfig = Field(default_factory=GitHubVerificationConfig)
    verification_labels: VerificationLabelsConfig = Field(
        default_factory=VerificationLabelsConfig
    )
    proxy_bias_flags: list[str] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if not _HAS_YAML:
        raise ImportError(
            "pyyaml is required to load config files. "
            "Install it with: pip install pyyaml"
        )
    with path.open(encoding="utf-8") as fh:
        return _yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_models_config() -> ModelsConfig:
    """Load and cache models.yaml. Injects API key from environment."""
    raw = _load_yaml(CONFIGS_DIR / "models.yaml")
    config = ModelsConfig(**raw)

    # Inject API key from environment — never hard-code credentials
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY")
    config.llm.api_key = api_key

    return config


def get_llm_config() -> LLMConfig:
    return get_models_config().llm


def get_embedding_config() -> EmbeddingConfig:
    return get_models_config().embeddings


def get_hyde_config() -> HyDEConfig:
    return get_models_config().hyde


@lru_cache(maxsize=1)
def get_verification_config() -> VerificationConfig:
    """Load and cache verification_rules.yaml."""
    raw = _load_yaml(CONFIGS_DIR / "verification_rules.yaml")
    return VerificationConfig(**raw)
