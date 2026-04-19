from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_skills_dir() -> str:
    return str(_PROJECT_ROOT / "skills")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_base_url: str = Field(default="http://localhost:11434/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(default="ollama", alias="OPENAI_API_KEY")

    model_requirement: str = Field(default="qwen2.5:14b", alias="MODEL_REQUIREMENT")
    model_config_plan: str = Field(default="qwen2.5:14b", alias="MODEL_CONFIG_PLAN")
    model_builder: str = Field(default="qwen2.5:14b", alias="MODEL_BUILDER")
    model_validator_soft: str = Field(default="qwen2.5:14b", alias="MODEL_VALIDATOR_SOFT")

    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    model_codegen: str = Field(default="minimax/minimax-m2.5:free", alias="MODEL_CODEGEN")

    git_repo_url: str = Field(default="", alias="GIT_REPO_URL")
    git_default_branch: str = Field(default="main", alias="GIT_DEFAULT_BRANCH")
    git_remote_name: str = Field(default="origin", alias="GIT_REMOTE_NAME")
    git_author_name: str = Field(default="InfraAi", alias="GIT_AUTHOR_NAME")
    git_author_email: str = Field(default="infra-ai@local", alias="GIT_AUTHOR_EMAIL")

    milvus_uri: str = Field(default="", alias="MILVUS_URI")
    skill_retrieval_mode: str = Field(default="filesystem", alias="SKILL_RETRIEVAL_MODE")

    skills_dir: str = Field(default_factory=_default_skills_dir, alias="SKILLS_DIR")


@lru_cache
def get_settings() -> Settings:
    return Settings()
