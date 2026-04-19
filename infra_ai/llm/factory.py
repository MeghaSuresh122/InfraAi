from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from infra_ai.config import get_settings

Role = Literal["requirement", "config_plan", "builder", "validator_soft", "codegen"]


def get_chat_model(role: Role) -> BaseChatModel:
    s = get_settings()
    if role == "codegen":
        if not s.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for code generation (OpenRouter). "
                "Set it in the environment or .env file."
            )
        return ChatOpenAI(
            base_url=s.openrouter_base_url,
            api_key=s.openrouter_api_key,
            model=s.model_codegen,
            temperature=0.2,
            default_headers={
                "HTTP-Referer": "https://github.com/infra-ai",
                "X-Title": "InfraAi",
            },
        )
    model_name = {
        "requirement": s.model_requirement,
        "config_plan": s.model_config_plan,
        "builder": s.model_builder,
        "validator_soft": s.model_validator_soft,
    }[role]
    base = s.openai_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return ChatOllama(
        base_url=base,
        model=model_name,
        temperature=0.1,
    )
