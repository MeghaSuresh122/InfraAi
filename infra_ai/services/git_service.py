import base64
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import jwt
import requests

from infra_ai.config import get_settings
from infra_ai.logging_config import get_logger

logger = get_logger(__name__)


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


def parse_github_repo(url: str) -> tuple[str, str] | None:
    """Parse owner and repo from GitHub URL, e.g., https://github.com/owner/repo -> ('owner', 'repo')."""
    if url.startswith("git@github.com:"):
        # Handle git@github.com:owner/repo.git
        parts = url[len("git@github.com:"):].split("/")
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1].rstrip(".git")
            return owner, repo
    parsed = urlparse(url)
    if parsed.hostname == "github.com":
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1].rstrip(".git")
            return owner, repo
    elif url.startswith("git@github.com:"):
        # git@github.com:owner/repo.git
        parts = url.split(":", 1)[1].split("/")
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1].rstrip(".git")
            return owner, repo
    return None


class GitService:
    """Clone, branch, commit, and push generated files."""

    def __init__(
        self,
        repo_url: str | None = None,
        default_branch: str | None = None,
        remote_name: str | None = None,
    ) -> None:
        s = get_settings()
        # Use explicit None check to allow empty strings
        self.repo_url = repo_url if repo_url is not None else s.git_repo_url
        self.default_branch = default_branch if default_branch is not None else s.git_default_branch
        self.remote_name = remote_name if remote_name is not None else s.git_remote_name
        self.author_name = s.git_author_name
        self.author_email = s.git_author_email
        self.github_client_id = s.github_app_client_id
        self.github_pem_path = s.github_app_pem_path
        self.github_installation_id = s.github_app_installation_id
        self._token_cache: dict[str, tuple[str, float]] = {}  # token: (token, expiry_time)

    def _generate_jwt(self) -> str:
        """Generate JWT for GitHub App authentication."""
        logger.debug("Generating JWT for GitHub App authentication")
        with open(self.github_pem_path, "rb") as f:
            private_key = f.read()
        now = int(time.time())
        payload = {
            "iat": now,
            "exp": now + 600,  # 10 minutes
            "iss": self.github_client_id,
        }
        jwt_token = jwt.encode(payload, private_key, algorithm="RS256")
        logger.debug("JWT token generated successfully")
        return jwt_token

    def _get_access_token(self) -> str:
        """Get cached or new installation access token."""
        cache_key = f"{self.github_client_id}_{self.github_installation_id}"
        if cache_key in self._token_cache:
            token, expiry = self._token_cache[cache_key]
            if time.time() < expiry:
                logger.debug("Using cached GitHub App access token")
                return token
        
        logger.info("Generating new GitHub App access token")
        jwt_token = self._generate_jwt()
        url = f"https://api.github.com/app/installations/{self.github_installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
        }
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            token = data["token"]
            expires_at = data["expires_at"]
            # Parse expires_at and cache for 1 hour less to be safe
            expiry_time = time.time() + 3600  # 1 hour
            self._token_cache[cache_key] = (token, expiry_time)
            logger.info("New access token generated and cached (expires in 1 hour)")
            return token
        except Exception as e:
            logger.error("Failed to get access token: %s", e)
            raise

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
        - Remote Git URL: writes under ``./output/...`` then uses GitHub API to create branch and commit files.
        """
        messages: list[str] = []
        branch_name = f"{branch_prefix}-{uuid4().hex[:8]}"
        logger.info("Starting push_files operation. Branch: %s, Files: %d", branch_name, len(files))
        
        out_dir = Path.cwd() / "output" / branch_name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Created output directory: %s", out_dir)
        
        for rel, content in files:
            path = out_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.debug("Wrote file: %s", rel)
        
        messages.append(f"Wrote files under {out_dir}")
        logger.info("All files written to output directory")

        if not self.repo_url:
            msg = "repo_url not set; skipped remote push and local project path."
            messages.append(msg)
            logger.warning(msg)
            return branch_name, messages

        if not is_remote_git_url(self.repo_url):
            logger.info("Local path detected: %s", self.repo_url)
            local_root = resolve_local_repo_root(self.repo_url)
            local_root.mkdir(parents=True, exist_ok=True)
            target = local_root / branch_name
            target.mkdir(parents=True, exist_ok=True)
            for rel, content in files:
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
                logger.debug("Wrote file to local path: %s", rel)
            msg = (
                f"Wrote files to local path {target} (no remote clone/push). "
                f"Mirror also at {out_dir}."
            )
            messages.append(msg)
            logger.info("Files written to local path: %s", target)
            return branch_name, messages

        # GitHub API push
        logger.info("Starting GitHub API push to: %s", self.repo_url)
        try:
            owner_repo = parse_github_repo(self.repo_url)
            if not owner_repo:
                raise ValueError(f"Could not parse owner/repo from {self.repo_url}")
            owner, repo = owner_repo
            logger.info("Pushing to GitHub repo: %s/%s", owner, repo)
            
            token = self._get_access_token()
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
            }

            # Get base branch SHA
            logger.debug("Fetching base branch SHA for: %s", self.default_branch)
            base_ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{self.default_branch}"
            response = requests.get(base_ref_url, headers=headers)
            response.raise_for_status()
            base_sha = response.json()["object"]["sha"]
            logger.debug("Base branch SHA: %s", base_sha[:8])

            # Create new branch
            logger.info("Creating new branch: %s", branch_name)
            new_ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
            ref_data = {
                "ref": f"refs/heads/{branch_name}",
                "sha": base_sha,
            }
            response = requests.post(new_ref_url, headers=headers, json=ref_data)
            response.raise_for_status()
            messages.append(f"Created branch {branch_name}")
            logger.info("Branch created successfully")

            # Create blobs
            logger.info("Creating %d blobs", len(files))
            blobs = []
            for rel, content in files:
                blob_url = f"https://api.github.com/repos/{owner}/{repo}/git/blobs"
                blob_data = {
                    "content": content,
                    "encoding": "utf-8",
                }
                response = requests.post(blob_url, headers=headers, json=blob_data)
                response.raise_for_status()
                blob_sha = response.json()["sha"]
                blobs.append({
                    "path": rel,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_sha,
                })
                logger.debug("Created blob for: %s (SHA: %s)", rel, blob_sha[:8])
            logger.info("All blobs created successfully")

            # Get base tree
            logger.debug("Fetching base tree SHA")
            base_commit_url = f"https://api.github.com/repos/{owner}/{repo}/git/commits/{base_sha}"
            response = requests.get(base_commit_url, headers=headers)
            response.raise_for_status()
            base_tree_sha = response.json()["tree"]["sha"]
            logger.debug("Base tree SHA: %s", base_tree_sha[:8])

            # Create new tree
            logger.info("Creating new tree with %d blobs", len(blobs))
            tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees"
            tree_data = {
                "base_tree": base_tree_sha,
                "tree": blobs,
            }
            response = requests.post(tree_url, headers=headers, json=tree_data)
            response.raise_for_status()
            new_tree_sha = response.json()["sha"]
            logger.debug("New tree created (SHA: %s)", new_tree_sha[:8])

            # Create commit
            logger.info("Creating commit for branch: %s", branch_name)
            commit_url = f"https://api.github.com/repos/{owner}/{repo}/git/commits"
            commit_data = {
                "message": f"InfraAi: add {branch_prefix} configs",
                "author": {
                    "name": self.author_name,
                    "email": self.author_email,
                },
                "parents": [base_sha],
                "tree": new_tree_sha,
            }
            response = requests.post(commit_url, headers=headers, json=commit_data)
            response.raise_for_status()
            new_commit_sha = response.json()["sha"]
            logger.debug("Commit created (SHA: %s)", new_commit_sha[:8])

            # Update branch ref
            logger.info("Updating branch reference to latest commit")
            update_ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch_name}"
            update_data = {"sha": new_commit_sha}
            response = requests.patch(update_ref_url, headers=headers, json=update_data)
            response.raise_for_status()
            msg = f"Pushed branch {branch_name} to GitHub."
            messages.append(msg)
            logger.info("Successfully pushed branch %s to GitHub", branch_name)

        except Exception as exc:  # noqa: BLE001
            logger.exception("GitHub API push failed")
            msg = f"GitHub API push failed (local copy still at {out_dir}): {exc}"
            messages.append(msg)

        return branch_name, messages

    def create_pull_request(self, head_branch: str, title: str, body: str = "") -> str | None:
        """
        Create a pull request from head_branch to the default branch.
        Returns the PR URL if successful, None if failed or not applicable.
        """
        if not self.repo_url or not is_remote_git_url(self.repo_url):
            logger.info("Skipping PR creation: not a remote GitHub repo")
            return None

        try:
            owner_repo = parse_github_repo(self.repo_url)
            if not owner_repo:
                raise ValueError(f"Could not parse owner/repo from {self.repo_url}")
            owner, repo = owner_repo
            logger.info("Creating PR for repo: %s/%s", owner, repo)

            token = self._get_access_token()
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
            }

            pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
            pr_data = {
                "title": title,
                "head": head_branch,
                "base": self.default_branch,
                "body": body,
            }
            response = requests.post(pr_url, headers=headers, json=pr_data)
            response.raise_for_status()
            pr_data = response.json()
            pr_html_url = pr_data["html_url"]
            logger.info("PR created successfully: %s", pr_html_url)
            return pr_html_url
        except Exception as exc:
            logger.exception("PR creation failed")
            return None
