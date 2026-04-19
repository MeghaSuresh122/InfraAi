from typing import Any

from pydantic import BaseModel, Field


class ConfigPlanItem(BaseModel):
    """One generated infrastructure artifact to implement."""

    id: str
    description: str
    type: str = Field(
        description="Artifact kind, e.g. terraform_eks_cluster, k8s_deployment, terraform_storage"
    )
    environment: str | None = None
    path_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConfigPlan(BaseModel):
    items: list[ConfigPlanItem]
