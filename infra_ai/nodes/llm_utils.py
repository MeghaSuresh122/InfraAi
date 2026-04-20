from __future__ import annotations

import json
import os
import re
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def mock_llm_enabled() -> bool:
    return os.environ.get("INFRA_AI_MOCK_LLM", "").lower() in ("1", "true", "yes")


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def invoke_structured(llm: BaseChatModel, prompt: str, model: type[T]) -> T:
    try:
        structured = llm.with_structured_output(model)
        out = structured.invoke([HumanMessage(content=prompt)])
        if isinstance(out, model):
            return out
        if isinstance(out, dict):
            return model.model_validate(out)
    except Exception:
        pass  # Fallback to manual parsing
    
    # Fallback: prompt the model to just output JSON and parse it manually
    fallback_prompt = prompt + "\n\nReturn ONLY a valid JSON object matching the requested schema. Do not include any other text."
    msg = llm.invoke([HumanMessage(content=fallback_prompt)])
    content = msg.content if hasattr(msg, "content") else str(msg)
    
    parsed = extract_json_object(str(content))
    if parsed:
        return model.model_validate(parsed)
        
    raise ValueError(f"Could not extract valid JSON from response: {content}")
