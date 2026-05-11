"""GitHub REST helpers for branch SHA and shallow clone URL (repo context)."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

from git import Repo

from infra_ai.services.git_service import GitService, is_remote_git_url, parse_github_repo

logger = logging.getLogger(__name__)


def fetch_github_branch_sha(
    repo_url: str,
    branch: str,
    git_svc: GitService | None = None,
) -> str | None:
    """Return commit SHA for branch tip on GitHub, or None if not applicable."""
    if not is_remote_git_url(repo_url):
        return None
    parsed = parse_github_repo(repo_url)
    if not parsed:
        return None
    owner, repo = parsed
    svc = git_svc or GitService(repo_url=repo_url, default_branch=branch)
    token = svc._get_access_token()  # noqa: SLF001
    import requests

    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    r = requests.get(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=60,
    )
    if r.status_code != 200:
        logger.warning("GitHub commits/%s failed: %s %s", branch, r.status_code, r.text[:200])
        return None
    return str(r.json().get("sha") or "")


def github_authenticated_clone_url(repo_url: str, git_svc: GitService) -> str | None:
    """HTTPS URL with embedded token for one-shot shallow clone."""
    parsed = parse_github_repo(repo_url)
    if not parsed:
        return None
    owner, repo = parsed
    token = git_svc._get_access_token()  # noqa: SLF001
    safe = quote(token, safe="")
    return f"https://x-access-token:{safe}@github.com/{owner}/{repo}.git"


def shallow_clone_branch(
    repo_url: str,
    branch: str,
    git_svc: GitService,
) -> Path:
    """Clone repo shallow (depth 1) to a temp directory; caller must delete tree."""
    clone_url = github_authenticated_clone_url(repo_url, git_svc)
    if not clone_url:
        raise ValueError("Cannot build clone URL for repo")
    tmp = Path(tempfile.mkdtemp(prefix="infra-ai-repoctx-"))
    target = tmp / "repo"
    logger.info("Shallow cloning branch %s into %s", branch, target)
    Repo.clone_from(clone_url, str(target), branch=branch, depth=1, single_branch=True)
    return tmp
