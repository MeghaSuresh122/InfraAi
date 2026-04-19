import sys
from pathlib import Path

import pytest

from infra_ai.services.git_service import GitService, is_remote_git_url, resolve_local_repo_root


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


def test_push_files_local_writes_under_branch(tmp_path: Path) -> None:
    svc = GitService(repo_url=str(tmp_path / "drop"))
    files = [("k8s/app.yaml", "apiVersion: v1\nkind: ConfigMap\n")]
    branch, messages = svc.push_files(files, branch_prefix="t1")
    written = list((tmp_path / "drop" / branch).rglob("*.yaml"))
    assert written
    assert written[0].read_text(encoding="utf-8").startswith("apiVersion")
    assert any("local path" in m.lower() for m in messages)
    assert any("no remote" in m.lower() for m in messages)


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
