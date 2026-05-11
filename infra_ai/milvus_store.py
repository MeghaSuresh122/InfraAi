"""
Milvus: skill chunks (optional) and repo-context RAG chunks.

Set MILVUS_URI (and optionally MILVUS_DATABASE, MILVUS_COLLECTION_REPO_CONTEXT); install infra-ai[milvus].
Repo context uses REPO_CONTEXT_VECTOR_STORE=auto|milvus|none (default auto).
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

from infra_ai.config import get_settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _expr_escape(value: str) -> str:
    """Escape for Milvus/VARCHAR filter strings inside double quotes."""
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


_DIM = 384  # all-MiniLM-L6-v2
_embedder = None  # sentence-transformers model cache


def _repo_context_collection() -> str:
    s = get_settings()
    name = (getattr(s, "milvus_collection_repo_context", None) or "").strip()
    if not name:
        name = (getattr(s, "milvus_collection", None) or "").strip()
    return name or "infra_ai_repo_context"


def milvus_enabled() -> bool:
    s = get_settings()
    return s.skill_retrieval_mode == "milvus" and bool(s.milvus_uri)


def repo_milvus_enabled() -> bool:
    s = get_settings()
    mode = (getattr(s, "repo_context_vector_store", None) or "auto").lower()
    if mode == "none":
        return False
    if mode == "milvus":
        return bool(s.milvus_uri)
    # auto
    return bool(s.milvus_uri)


def _client():
    try:
        from pymilvus import MilvusClient
    except ImportError:
        return None
    s = get_settings()
    if not s.milvus_uri:
        return None
    db = (getattr(s, "milvus_database", None) or "").strip()
    if db:
        return MilvusClient(uri=s.milvus_uri, db_name=db)
    return MilvusClient(uri=s.milvus_uri)


def _ensure_collection(client: Any) -> None:
    coll = _repo_context_collection()
    if client.has_collection(coll):
        return
    try:
        from pymilvus import DataType, MilvusClient
    except ImportError:
        return
    try:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=_DIM)
        idx = client.prepare_index_params()
        idx.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")
        client.create_collection(collection_name=coll, schema=schema, index_params=idx)
    except Exception as e:  # noqa: BLE001
        logger.warning("Milvus create_collection failed, falling back to no collection: %s", e)


def _embed(texts: list[str]) -> list[list[float]]:
    global _embedder  # noqa: PLW0603
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed; Milvus embed disabled")
        return []
    if _embedder is None:
        model_name = getattr(get_settings(), "repo_context_embedding_model", None) or (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        _embedder = SentenceTransformer(model_name)
    return _embedder.encode(texts, normalize_embeddings=True).tolist()


def query_skill_chunks(_query: str, _artifact_type: str, _top_k: int = 5) -> list[str]:
    if not milvus_enabled():
        return []
    try:
        import pymilvus  # noqa: F401
    except ImportError:
        return []
    return []


def delete_repo_context_for_branch(repo_key: str, base_branch: str) -> None:
    client = _client()
    coll = _repo_context_collection()
    if not client or not client.has_collection(coll):
        return
    _ensure_collection(client)
    expr = (
        f'repo_key == "{_expr_escape(repo_key)}" and base_branch == "{_expr_escape(base_branch)}"'
    )
    try:
        client.delete(collection_name=coll, filter=expr)
    except Exception as e:  # noqa: BLE001
        logger.warning("Milvus delete failed: %s", e)


def upsert_repo_context_chunks(
    repo_key: str,
    base_branch: str,
    head_commit: str,
    chunks: list[dict[str, Any]],
) -> int:
    """
    Each chunk: path, chunk_type, text.
    Replaces vectors for this repo_key+base_branch (caller should delete stale first if head changed).
    """
    if not repo_milvus_enabled() or not chunks:
        return 0
    client = _client()
    if not client:
        return 0
    coll = _repo_context_collection()
    _ensure_collection(client)
    if not client.has_collection(coll):
        return 0
    texts = [c.get("text") or "" for c in chunks]
    vectors = _embed(texts)
    if not vectors or len(vectors) != len(chunks):
        return 0
    rows: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        path = (ch.get("path") or "")[:500]
        ctype = (ch.get("chunk_type") or "file_excerpt")[:64]
        raw = f"{repo_key}|{base_branch}|{head_commit}|{path}|{i}|{ch.get('text', '')[:200]}"
        cid = hashlib.sha256(raw.encode()).hexdigest()[:32]
        rows.append(
            {
                "id": cid,
                "vector": vectors[i],
                # dynamic fields (Milvus $meta) for filter + retrieval
                "repo_key": repo_key[:500],
                "base_branch": base_branch[:200],
                "head_commit": head_commit[:64],
                "path": path,
                "chunk_type": ctype,
                "text": (ch.get("text") or "")[:8000],
                "updated_at": int(time.time()),
            }
        )
    try:
        client.insert(collection_name=coll, data=rows)
        return len(rows)
    except Exception as e:  # noqa: BLE001
        logger.exception("Milvus insert failed: %s", e)
        return 0


def query_repo_context_chunks(
    query: str,
    repo_key: str,
    base_branch: str,
    head_commit: str,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    if not repo_milvus_enabled():
        return []
    client = _client()
    if not client:
        return []
    coll = _repo_context_collection()
    if not client.has_collection(coll):
        return []
    qvec = _embed([query[:4000]])
    if not qvec:
        return []
    try:
        res = client.search(
            collection_name=coll,
            data=qvec,
            limit=top_k,
            filter=(
                f'repo_key == "{_expr_escape(repo_key)}" and base_branch == "{_expr_escape(base_branch)}" '
                f'and head_commit == "{_expr_escape(head_commit[:64])}"'
            ),
            output_fields=["path", "chunk_type", "text"],
        )
        out: list[dict[str, Any]] = []
        for hits in res or []:
            for hit in hits:
                entity = hit.get("entity", {}) or {}
                out.append(
                    {
                        "path": entity.get("path", ""),
                        "chunk_type": entity.get("chunk_type", ""),
                        "text": (entity.get("text") or "")[:2000],
                        "score": float(hit.get("distance", 0.0)),
                    }
                )
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("Milvus search failed: %s", e)
        return []
