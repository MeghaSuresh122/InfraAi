from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from git import Actor, Repo

from infra_ai.config import get_settings


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
        Returns (branch_name, messages). If no repo_url, writes to ./output only.
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
            messages.append("GIT_REPO_URL not set; skipped git clone/push.")
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
