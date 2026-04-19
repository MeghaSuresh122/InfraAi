"""
Optional Milvus-backed skill chunk retrieval.

Set SKILL_RETRIEVAL_MODE=milvus and MILVUS_URI; install infra-ai[milvus].
When pymilvus is unavailable, operations no-op and callers should use filesystem skills only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from infra_ai.config import get_settings

if TYPE_CHECKING:
    pass


def milvus_enabled() -> bool:
    s = get_settings()
    return s.skill_retrieval_mode == "milvus" and bool(s.milvus_uri)


def query_skill_chunks(_query: str, _artifact_type: str, _top_k: int = 5) -> list[str]:
    if not milvus_enabled():
        return []
    try:
        import pymilvus  # noqa: F401
    except ImportError:
        return []
    # Placeholder: wire collection search when embeddings pipeline exists.
    return []
