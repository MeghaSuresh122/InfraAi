from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from git import Actor, Repo

from infra_ai.config import get_settings


def is_remote_git_url(url: str) -> bool:
    """
    True if repo_url should be cloned/pushed as a remote Git repository.
    Filesystem paths (including Windows drive paths and ./relative dirs) return False.
    """
    u = (url or "").strip()
    if not u:
        return False
    if u.lower().startswith("git@"):
        return True
    parsed = urlparse(u)
    if parsed.scheme:
        return parsed.scheme.lower() in ("http", "https", "git", "ssh")
    return False


def resolve_local_repo_root(url: str) -> Path:
    """Resolve ``repo_url`` to an absolute directory when it is a local (non-remote) target."""
    u = url.strip()
    parsed = urlparse(u)
    if parsed.scheme.lower() == "file":
        raw = parsed.path or "/"
        if os.name == "nt" and len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        return Path(raw).expanduser().resolve()
    return Path(u).expanduser().resolve()


class GitService:
    """Clone, branch, commit, and push generated files."""

    def __init__(
        self,
        repo_url: str | None = None,
        default_branch: str | None = None,
        remote_name: str | None = None,
    ) -> None:
        s = get_settings()
        self.repo_url = repo_url or s.git_repo_url
        self.default_branch = default_branch or s.git_default_branch
        self.remote_name = remote_name or s.git_remote_name
        self.author = Actor(s.git_author_name, s.git_author_email)

    def push_files(
        self,
        files: list[tuple[str, str]],
        branch_prefix: str,
    ) -> tuple[str, list[str]]:
        """
        Returns (branch_name, messages).

        - No ``repo_url``: writes under ``./output/<branch_name>/`` only (no push).
        - Local path (not a remote Git URL): writes under ``<repo_url>/<branch_name>/`` only
          (no clone, no GitHub push).
        - Remote Git URL: writes under ``./output/...`` then clones, commits, and pushes a new branch.
        """
        messages: list[str] = []
        branch_name = f"{branch_prefix}-{uuid4().hex[:8]}"
        out_dir = Path.cwd() / "output" / branch_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for rel, content in files:
            path = out_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        messages.append(f"Wrote files under {out_dir}")

        if not self.repo_url:
            messages.append("repo_url not set; skipped remote push and local project path.")
            return branch_name, messages

        if not is_remote_git_url(self.repo_url):
            local_root = resolve_local_repo_root(self.repo_url)
            local_root.mkdir(parents=True, exist_ok=True)
            target = local_root / branch_name
            target.mkdir(parents=True, exist_ok=True)
            for rel, content in files:
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            messages.append(
                f"Wrote files to local path {target} (no remote clone/push). "
                f"Mirror also at {out_dir}."
            )
            return branch_name, messages

        tmp = Path(tempfile.mkdtemp(prefix="infra-ai-git-"))
        try:
            repo = Repo.clone_from(self.repo_url, tmp, branch=self.default_branch, depth=1)
            new_branch = repo.create_head(branch_name)
            new_branch.checkout()
            for rel, content in files:
                dest = tmp / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            repo.index.add([f[0] for f in files])
            repo.index.commit(
                f"InfraAi: add {branch_prefix} configs",
                author=self.author,
                committer=self.author,
            )
            origin = repo.remote(name=self.remote_name)
            origin.push(refspec=f"{branch_name}:{branch_name}")
            messages.append(f"Pushed branch {branch_name} to origin.")
        except Exception as exc:  # noqa: BLE001
            messages.append(f"Git push failed (local copy still at {out_dir}): {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        return branch_name, messages
