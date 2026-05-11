"""Approximate token budgeting for repo RAG text blocks."""

from __future__ import annotations

from typing import Any


def approx_tokens(text: str) -> int:
    """Rough heuristic: ~4 chars per token for English/code mix."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_bounded_repo_rag_text(
    sections: list[tuple[str, str]],
    max_tokens: int = 3000,
) -> tuple[str, dict[str, Any]]:
    """
    sections: list of (title, body). Preserves section order; trims bodies from end of list
    when over budget. Within oversized single body, hard-truncate.
    """
    meta: dict[str, Any] = {"dropped_sections": [], "truncated": []}
    parts: list[str] = []
    budget = max_tokens

    for title, body in sections:
        header = f"## {title}\n"
        h_cost = approx_tokens(header)
        if h_cost >= budget:
            meta["dropped_sections"].append(title)
            break
        budget -= h_cost
        b = body or ""
        bt = approx_tokens(b)
        if bt <= budget:
            parts.append(header + b)
            budget -= bt
            continue
        # Truncate body to fit
        char_budget = max(0, budget * 4 - 50)
        trimmed = b[:char_budget] + "\n...[truncated]..."
        parts.append(header + trimmed)
        meta["truncated"].append(title)
        budget = 0
        break

    out = "\n\n".join(parts)
    return out, meta
