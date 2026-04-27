from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from infra_ai.config import get_settings

Role = Literal["requirement", "config_plan", "builder", "validator_soft", "codegen"]


def get_chat_model(role: Role) -> BaseChatModel:
    s = get_settings()
    provider = {
        "requirement": s.llm_provider_requirement,
        "config_plan": s.llm_provider_config_plan,
        "builder": s.llm_provider_builder,
        "validator_soft": s.llm_provider_validator_soft,
        "codegen": s.llm_provider_codegen,
    }[role]

    model_name = {
        "requirement": s.model_requirement,
        "config_plan": s.model_config_plan,
        "builder": s.model_builder,
        "validator_soft": s.model_validator_soft,
        "codegen": s.model_codegen,
    }[role]

    if provider == "openrouter":
        if not s.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouter provider. "
                "Set it in the environment or .env file."
            )
        base_url = s.openrouter_base_url
        api_key = s.openrouter_api_key
        model_kwargs = {"extra_body": {"max_tokens": 2048}}
        default_headers = {
            "HTTP-Referer": "https://github.com/infra-ai",
            "X-Title": "InfraAi",
        }
    elif provider == "ollama":
        base_url = s.openai_base_url
        api_key = s.openai_api_key
        model_kwargs = {}
        default_headers = None
    elif provider == "groq":
        if not s.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required for Groq provider. "
                "Set it in the environment or .env file."
            )
        base_url = s.groq_base_url
        api_key = s.groq_api_key
        model_kwargs = {}
        default_headers = None
    elif provider == "gemini":
        if not s.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for Gemini provider. "
                "Set it in the environment or .env file."
            )
        # Gemini uses a different client
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=s.gemini_api_key,
            temperature=0.2 if role == "codegen" else 0.1,
            max_tokens=2048,
            max_retries=4
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0.2 if role == "codegen" else 0.1,
        max_tokens=2048,
        max_completion_tokens=2048,
        model_kwargs=model_kwargs,
        default_headers=default_headers,
    )
