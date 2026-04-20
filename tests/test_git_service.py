import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infra_ai.services.git_service import GitService, is_remote_git_url, parse_github_repo, resolve_local_repo_root


@pytest.mark.parametrize(
    ("url", "expected_remote"),
    [
        ("", False),
        ("https://github.com/org/repo.git", True),
        ("git@github.com:org/repo.git", True),
        ("./local-out", False),
        ("C:/infra/local", False),
        ("file:///C:/tmp/out", False),
    ],
)
def test_is_remote_git_url(url: str, expected_remote: bool) -> None:
    assert is_remote_git_url(url) is expected_remote


def test_parse_github_repo() -> None:
    assert parse_github_repo("https://github.com/owner/repo") == ("owner", "repo")
    assert parse_github_repo("https://github.com/owner/repo.git") == ("owner", "repo")
    assert parse_github_repo("git@github.com:owner/repo.git") == ("owner", "repo")
    assert parse_github_repo("https://gitlab.com/owner/repo") is None


def test_push_files_local_writes_under_branch(tmp_path: Path) -> None:
    svc = GitService(repo_url=str(tmp_path / "drop"))
    files = [("k8s/app.yaml", "apiVersion: v1\nkind: ConfigMap\n")]
    branch, messages = svc.push_files(files, branch_prefix="t1")
    written = list((tmp_path / "drop" / branch).rglob("*.yaml"))
    assert written
    assert written[0].read_text(encoding="utf-8").startswith("apiVersion")
    assert any("local path" in m.lower() for m in messages)
    assert any("no remote" in m.lower() for m in messages)


@patch("infra_ai.services.git_service.requests")
@patch("infra_ai.services.git_service.get_settings")
def test_push_files_remote_github_api(mock_get_settings: MagicMock, mock_requests: MagicMock) -> None:
    # Mock settings
    mock_settings = MagicMock()
    mock_settings.git_repo_url = "https://github.com/owner/repo"
    mock_settings.git_default_branch = "main"
    mock_settings.git_remote_name = "origin"
    mock_settings.git_author_name = "InfraAi"
    mock_settings.git_author_email = "infra-ai@local"
    mock_settings.github_app_client_id = "client_id"
    mock_settings.github_app_pem_path = "/path/to/pem"
    mock_settings.github_app_installation_id = "123"
    mock_get_settings.return_value = mock_settings

    # Mock the GitHub API responses in order of calls
    mock_responses = [
        MagicMock(json=MagicMock(return_value={"object": {"sha": "base_sha"}})),  # GET base ref
        MagicMock(),  # POST create ref (no json needed)
        MagicMock(json=MagicMock(return_value={"sha": "blob_sha"})),  # POST blob
        MagicMock(json=MagicMock(return_value={"tree": {"sha": "tree_sha"}})),  # GET commit
        MagicMock(json=MagicMock(return_value={"sha": "new_tree_sha"})),  # POST tree
        MagicMock(json=MagicMock(return_value={"sha": "new_commit_sha"})),  # POST commit
        MagicMock(),  # PATCH update ref (no json needed)
    ]
    mock_requests.get.side_effect = [mock_responses[0], mock_responses[3]]  # GET calls
    mock_requests.post.side_effect = [mock_responses[1], mock_responses[2], mock_responses[4], mock_responses[5]]  # POST calls
    mock_requests.patch.side_effect = [mock_responses[6]]  # PATCH call

    svc = GitService(repo_url="https://github.com/owner/repo")
    files = [("main.tf", 'terraform { required_version = ">= 1.5.0" }')]

    # Mock the JWT and token
    with patch.object(svc, "_generate_jwt", return_value="jwt_token"), \
         patch.object(svc, "_get_access_token", return_value="access_token"):
        branch, messages = svc.push_files(files, branch_prefix="tf")

    assert "Pushed branch" in " ".join(messages)
    # Verify API calls were made
    assert mock_requests.get.call_count == 2  # Get base ref, get commit
    assert mock_requests.post.call_count == 4  # Create blob, create ref, create tree, create commit
    assert mock_requests.patch.call_count == 1  # Update ref


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file URI")
def test_resolve_local_file_uri_windows_path() -> None:
    p = resolve_local_repo_root("file:///C:/Windows")
    assert p.is_absolute()
    assert "Windows" in str(p)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file URI")
def test_resolve_local_file_uri_posix() -> None:
    p = resolve_local_repo_root("file:///tmp")
    assert p.is_absolute()
    assert p.name == "tmp"
