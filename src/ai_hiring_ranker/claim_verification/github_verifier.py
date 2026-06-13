"""
GitHub Verifier — checks skill claims against a candidate's GitHub profile.

Strategy:
  1. Parse the GitHub username from the profile URL.
  2. Fetch public repos via the GitHub REST API (unauthenticated = 60 req/hr;
     set GITHUB_TOKEN for 5000 req/hr).
  3. For each skill, look for evidence across:
     - Repo languages (most reliable signal)
     - Repo names and descriptions
     - README content (fetched lazily, max 3 repos)
     - File extensions in repo trees (Dockerfile, *.py, requirements.txt, etc.)
     - Recent commits and test/config files when enabled in verification_rules.yaml
  4. Score recency based on pushed_at date and apply cutoff penalties.
  5. Return skill-tagged EvidenceItem objects.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from ..config import VerificationConfig, get_verification_config
from .schemas import EvidenceItem, EvidenceSource
from .utils import apply_recency_penalty

logger = logging.getLogger(__name__)

MAX_REPOS = 10
MAX_README_REPOS = 3
MAX_COMMIT_REPOS = 3
MAX_COMMITS_PER_REPO = 5

_SKILL_GITHUB_SIGNALS: dict[str, dict] = {
    "Python": {"languages": ["Python"], "extensions": [".py"], "files": ["requirements.txt", "setup.py", "pyproject.toml"]},
    "Fastapi": {"languages": ["Python"], "extensions": [".py"], "files": ["main.py"], "keywords": ["fastapi", "from fastapi"]},
    "Docker": {"languages": [], "extensions": [], "files": ["Dockerfile", "docker-compose.yml", ".dockerignore"]},
    "Kubernetes": {"languages": [], "extensions": [".yaml", ".yml"], "files": ["deployment.yaml", "values.yaml"], "keywords": ["kubectl", "kubernetes", "k8s"]},
    "Machine Learning": {"languages": ["Python", "Jupyter Notebook"], "extensions": [".py", ".ipynb"], "files": ["requirements.txt"], "keywords": ["sklearn", "torch", "tensorflow", "xgboost", "model"]},
    "Deep Learning": {"languages": ["Python", "Jupyter Notebook"], "extensions": [".py", ".ipynb"], "keywords": ["pytorch", "tensorflow", "keras", "neural"]},
    "Sql": {"languages": ["PLpgSQL", "SQL"], "extensions": [".sql"], "keywords": ["select", "create table", "sqlalchemy", "psycopg"]},
    "Nosql": {"languages": [], "extensions": [], "keywords": ["mongodb", "pymongo", "redis", "dynamodb"]},
    "Rest Api": {"languages": ["Python", "JavaScript", "TypeScript"], "keywords": ["api", "endpoint", "router", "flask", "fastapi", "express"]},
    "Cloud": {"languages": [], "extensions": [".tf", ".yaml"], "files": ["serverless.yml", "terraform.tf"], "keywords": ["aws", "gcp", "azure", "s3", "lambda", "cloud"]},
    "Ci/Cd": {"languages": [], "extensions": [".yml"], "files": [".github/workflows", "Jenkinsfile", ".gitlab-ci.yml"]},
    "Testing": {"languages": ["Python"], "extensions": [".py"], "files": ["conftest.py"], "keywords": ["pytest", "unittest", "test_", "def test"]},
    "Embeddings": {"languages": ["Python"], "extensions": [".py", ".ipynb"], "keywords": ["embedding", "faiss", "sentence_transformers", "openai.embeddings"]},
    "Retrieval": {"languages": ["Python"], "extensions": [".py"], "keywords": ["faiss", "bm25", "retrieval", "vector search", "elasticsearch"]},
    "Llm": {"languages": ["Python"], "extensions": [".py", ".ipynb"], "keywords": ["openai", "anthropic", "llm", "gpt", "langchain", "prompt"]},
    "Langchain": {"languages": ["Python"], "extensions": [".py"], "keywords": ["langchain", "langgraph", "from langchain"]},
    "Model Evaluation": {"languages": ["Python"], "extensions": [".py", ".ipynb"], "keywords": ["accuracy", "f1_score", "roc_auc", "precision", "recall", "evaluate"]},
    "Data Engineering": {"languages": ["Python"], "extensions": [".py"], "keywords": ["pipeline", "etl", "airflow", "spark", "dbt"]},
    "Spark": {"languages": ["Python", "Scala"], "extensions": [".py", ".scala"], "keywords": ["pyspark", "spark", "rdd", "dataframe"]},
    "Nlp": {"languages": ["Python"], "extensions": [".py", ".ipynb"], "keywords": ["spacy", "nltk", "tokenize", "ner", "transformers", "bert"]},
    "Git": {"languages": [], "extensions": [], "keywords": []},
}

_TEST_FILE_HINTS = (
    "test_",
    "_test.py",
    "tests/",
    "conftest.py",
    ".github/workflows",
    "pytest.ini",
    "tox.ini",
)


def _github_get(url: str, token: Optional[str]) -> Optional[dict | list]:
    """Make a GET request to the GitHub API. Returns parsed JSON or None on error."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ai-hiring-ranker/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.debug("GitHub 404: %s", url)
        elif exc.code == 403:
            logger.warning("GitHub rate limit or auth error: %s", url)
        else:
            logger.warning("GitHub HTTP %d: %s", exc.code, url)
        return None
    except Exception as exc:
        logger.warning("GitHub request failed (%s): %s", type(exc).__name__, url)
        return None


def _recency_years(date_str: Optional[str]) -> Optional[float]:
    """Convert an ISO date string to 'years ago'."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        return round(delta.days / 365.25, 1)
    except Exception:
        return None


def _username_from_url(github_url: str) -> Optional[str]:
    """Extract GitHub username from a profile or repo URL."""
    match = re.search(r"github\.com/([A-Za-z0-9_.\-]+)", github_url)
    if not match:
        return None
    parts = github_url.rstrip("/").split("github.com/")[-1].split("/")
    return parts[0] if parts else None


def _repo_has_test_files(repo_files: set[str]) -> list[str]:
    return [path for path in repo_files if any(hint in path for hint in _TEST_FILE_HINTS)]


def _fetch_recent_commits(
    owner: str,
    repo_name: str,
    token: Optional[str],
    config: VerificationConfig,
) -> list[dict]:
    if not config.github.check_commits:
        return []

    commits_url = (
        f"https://api.github.com/repos/{owner}/{repo_name}/commits"
        f"?per_page={MAX_COMMITS_PER_REPO}"
    )
    commits = _github_get(commits_url, token)
    if not isinstance(commits, list):
        return []
    return commits


def verify_skills_via_github(
    github_url: str,
    skills: list[str],
    candidate_id: str,
    config: Optional[VerificationConfig] = None,
) -> tuple[list[EvidenceItem], list[str]]:
    """
    Check skill claims against a GitHub profile.

    Args:
        github_url:   The candidate's GitHub profile URL.
        skills:       List of normalised skill names to check.
        candidate_id: Used for logging only.
        config:       Optional verification config override.

    Returns:
        (evidence_items, error_notes)
    """
    cfg = config or get_verification_config()
    token = os.getenv("GITHUB_TOKEN")
    evidence: list[EvidenceItem] = []
    errors: list[str] = []

    username = _username_from_url(github_url)
    if not username:
        errors.append(f"Could not parse GitHub username from URL: {github_url}")
        return evidence, errors

    repos_url = f"https://api.github.com/users/{username}/repos?per_page={MAX_REPOS}&sort=pushed"
    repos = _github_get(repos_url, token)
    if not isinstance(repos, list):
        errors.append(f"GitHub API returned no repos for {username}")
        return evidence, errors

    logger.info("[%s] GitHub: found %d repos for %s", candidate_id, len(repos), username)

    active_repos = [repo for repo in repos[:MAX_REPOS] if not repo.get("fork")]
    all_languages: set[str] = set()
    repo_names: list[str] = []
    repo_descriptions: list[str] = []
    pushed_dates: list[str] = []
    repo_files: set[str] = set()
    commit_evidence: list[EvidenceItem] = []
    test_evidence: list[EvidenceItem] = []

    for repo in active_repos:
        langs = repo.get("language") or ""
        if langs:
            all_languages.add(langs)
        repo_names.append((repo.get("name") or "").lower())
        repo_descriptions.append((repo.get("description") or "").lower())
        pushed_dates.append(repo.get("pushed_at") or "")

    most_recent_push = max(pushed_dates, default=None) if pushed_dates else None
    recency = _recency_years(most_recent_push)

    for repo in active_repos[:5]:
        lang_url = repo.get("languages_url", "")
        if lang_url:
            repo_langs = _github_get(lang_url, token)
            if isinstance(repo_langs, dict):
                all_languages.update(repo_langs.keys())

    readme_texts: list[str] = []
    if cfg.github.check_readme:
        for repo in active_repos[:MAX_README_REPOS]:
            owner = repo.get("owner", {}).get("login", username)
            name = repo.get("name", "")
            readme_url = f"https://api.github.com/repos/{owner}/{name}/readme"
            readme_data = _github_get(readme_url, token)
            if isinstance(readme_data, dict):
                content = readme_data.get("content", "")
                try:
                    decoded = base64.b64decode(content).decode("utf-8", errors="replace").lower()
                    readme_texts.append(decoded[:3000])
                except Exception:
                    pass

    if cfg.github.check_file_types or cfg.github.check_tests:
        for repo in active_repos[:5]:
            owner = repo.get("owner", {}).get("login", username)
            name = repo.get("name", "")
            sha = repo.get("default_branch", "main")
            tree_url = f"https://api.github.com/repos/{owner}/{name}/git/trees/{sha}?recursive=1"
            tree_data = _github_get(tree_url, token)
            repo_paths: set[str] = set()
            if isinstance(tree_data, dict):
                for item in tree_data.get("tree", [])[:200]:
                    repo_paths.add((item.get("path") or "").lower())
            repo_files.update(repo_paths)

            if cfg.github.check_tests:
                matched_tests = _repo_has_test_files(repo_paths)
                if matched_tests:
                    test_evidence.append(
                        EvidenceItem(
                            source=EvidenceSource.GITHUB_FILE,
                            url=f"https://github.com/{owner}/{name}",
                            skill="Testing",
                            snippet=f"Test/CI files found: {', '.join(matched_tests[:4])}",
                            file_path=matched_tests[0],
                            recency_years=_recency_years(repo.get("pushed_at")),
                            relevance_score=0.8,
                        )
                    )

            if cfg.github.check_commits:
                commits = _fetch_recent_commits(owner, name, token, cfg)
                for commit in commits[:2]:
                    commit_sha = commit.get("sha", "")
                    commit_msg = (
                        commit.get("commit", {}).get("message", "") or ""
                    ).lower()
                    commit_date = commit.get("commit", {}).get("author", {}).get("date")
                    commit_evidence.append(
                        EvidenceItem(
                            source=EvidenceSource.GITHUB_COMMIT,
                            url=f"https://github.com/{owner}/{name}/commit/{commit_sha}",
                            skill=None,
                            snippet=f"Recent commit in {name}: {commit_msg[:120]}",
                            commit_sha=commit_sha,
                            recency_years=_recency_years(commit_date),
                            relevance_score=0.7,
                        )
                    )

    combined_text = " ".join(repo_names + repo_descriptions + readme_texts)

    for skill in skills:
        signals = _SKILL_GITHUB_SIGNALS.get(skill, {})
        items_for_skill: list[EvidenceItem] = []

        if cfg.github.check_file_types:
            lang_targets = signals.get("languages", [])
            matched_langs = [lang for lang in lang_targets if lang in all_languages]
            if matched_langs:
                items_for_skill.append(
                    EvidenceItem(
                        source=EvidenceSource.GITHUB_REPO,
                        url=f"https://github.com/{username}",
                        skill=skill,
                        snippet=f"GitHub languages detected: {', '.join(matched_langs)}",
                        recency_years=recency,
                        relevance_score=0.8,
                    )
                )

            file_targets = signals.get("files", [])
            matched_files = [
                file_name
                for file_name in file_targets
                if any(file_name.lower() in repo_path for repo_path in repo_files)
            ]
            if matched_files:
                items_for_skill.append(
                    EvidenceItem(
                        source=EvidenceSource.GITHUB_FILE,
                        url=f"https://github.com/{username}",
                        skill=skill,
                        snippet=f"Found project files: {', '.join(matched_files)}",
                        file_path=matched_files[0],
                        recency_years=recency,
                        relevance_score=0.85,
                    )
                )

        kw_targets = signals.get("keywords", [])
        matched_kws = [kw for kw in kw_targets if kw in combined_text]
        if matched_kws:
            items_for_skill.append(
                EvidenceItem(
                    source=EvidenceSource.GITHUB_REPO,
                    url=f"https://github.com/{username}",
                    skill=skill,
                    snippet=f"Keywords found in repos/READMEs: {', '.join(matched_kws[:4])}",
                    recency_years=recency,
                    relevance_score=0.7,
                )
            )

        if skill == "Git" and repos:
            items_for_skill.append(
                EvidenceItem(
                    source=EvidenceSource.GITHUB_REPO,
                    url=f"https://github.com/{username}",
                    skill=skill,
                    snippet=f"Active GitHub profile with {len(repos)} public repos.",
                    recency_years=recency,
                    relevance_score=0.9,
                )
            )

        for item in items_for_skill:
            evidence.append(
                apply_recency_penalty(item, cfg.github.recency_cutoff_years)
            )

    for item in test_evidence + commit_evidence:
        if item.skill is None:
            for skill in skills:
                if skill.lower() in item.snippet.lower():
                    item = item.model_copy(update={"skill": skill})
                    break
        evidence.append(apply_recency_penalty(item, cfg.github.recency_cutoff_years))

    return evidence, errors
