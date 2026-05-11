"""Split repo file texts into Milvus-friendly chunks."""

from __future__ import annotations

from typing import Any


def chunk_text(path: str, text: str, chunk_size: int = 900, overlap: int = 120) -> list[dict[str, Any]]:
    t = text or ""
    if not t.strip():
        return []
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(t):
        piece = t[i : i + chunk_size]
        out.append(
            {
                "path": path,
                "chunk_type": "file_excerpt",
                "text": f"FILE:{path}\n{piece}",
            }
        )
        i += chunk_size - overlap
        if i <= 0:
            break
    return out


def summary_chunks(repo_summary: dict[str, Any], reusable_modules: dict[str, Any]) -> list[dict[str, Any]]:
    import json

    chunks: list[dict[str, Any]] = []
    s = json.dumps(repo_summary, ensure_ascii=False, default=str)[:12000]
    chunks.append({"path": "", "chunk_type": "summary", "text": f"REPO_SUMMARY\n{s}"})
    m = json.dumps(reusable_modules, ensure_ascii=False, default=str)[:12000]
    chunks.append({"path": "", "chunk_type": "module", "text": f"REUSABLE_MODULES\n{m}"})
    return chunks
