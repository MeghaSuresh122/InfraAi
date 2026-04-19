from typing import Any

from pydantic import BaseModel, Field


class ReviewFieldsResume(BaseModel):
    """Human edits the config_fields_output JSON."""

    config_fields: dict[str, Any] = Field(
        description="Same shape as agent output: keys map to {value, agent_generated, confidence_score}"
    )


class RepoConfirmResume(BaseModel):
    """Human confirms codegen and may override repo URL / branch."""

    confirm: bool = True
    repo_url: str | None = None
    target_branch: str | None = None


class ContinueNextResume(BaseModel):
    """After push: continue with next config file."""

    continue_next: bool = Field(description="If true, process next item in config_plan")
