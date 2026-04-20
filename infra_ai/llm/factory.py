from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from infra_ai.config import get_settings

Role = Literal["requirement", "config_plan", "builder", "validator_soft", "codegen"]


def get_chat_model(role: Role) -> BaseChatModel:
    s = get_settings()
    if not s.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required for all OpenRouter agents. "
            "Set it in the environment or .env file."
        )
    model_name = {
        "requirement": s.model_requirement,
        "config_plan": s.model_config_plan,
        "builder": s.model_builder,
        "validator_soft": s.model_validator_soft,
        "codegen": s.model_codegen,
    }[role]
    return ChatOpenAI(
        base_url=s.openrouter_base_url,
        api_key=s.openrouter_api_key,
        model=model_name,
        temperature=0.2 if role == "codegen" else 0.1,
        max_tokens=2048,
        max_completion_tokens=2048,
        model_kwargs={"extra_body": {"max_tokens": 2048}},
        default_headers={
            "HTTP-Referer": "https://github.com/infra-ai",
            "X-Title": "InfraAi",
        },
    )
