from typing import Any

from pydantic import BaseModel, Field, field_validator


class ConfigFieldValue(BaseModel):
    value: Any
    agent_generated: bool = False
    confidence_score: float = Field(ge=0.0, le=9.9)

    @field_validator("confidence_score")
    @classmethod
    def cap_confidence(cls, v: float) -> float:
        return min(9.9, max(0.0, v))


class ConfigFieldsEnvelope(BaseModel):
    """Map of field name -> metadata envelope."""

    fields: dict[str, ConfigFieldValue]

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any]) -> "ConfigFieldsEnvelope":
        out: dict[str, ConfigFieldValue] = {}
        for k, v in data.items():
            if isinstance(v, dict) and "value" in v:
                out[k] = ConfigFieldValue.model_validate(v)
            else:
                out[k] = ConfigFieldValue(value=v, agent_generated=False, confidence_score=9.9)
        return cls(fields=out)

    def to_flat_dict(self) -> dict[str, Any]:
        return {name: fv.model_dump() for name, fv in self.fields.items()}
