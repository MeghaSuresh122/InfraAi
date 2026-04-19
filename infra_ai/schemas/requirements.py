from typing import Any

from pydantic import BaseModel, Field


class RequirementAnalysis(BaseModel):
    """Structured output from Requirement Analysis Agent."""

    application_type: str | None = None
    application_tech: list[str] = Field(default_factory=list)
    expected_cpu: str | None = None
    expected_memory: str | None = None
    environments: list[str] = Field(default_factory=list)
    region: str | None = None
    cloud_provider: str | None = None
    scaling: dict[str, Any] | None = None
    notes: str | None = None
    extra_configs: dict[str, Any] = Field(default_factory=dict)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=False)
