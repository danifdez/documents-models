"""Model configuration shared by task implementations."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelSpec:
    """Resolves to (model_path, lora_path, lora_scale) tuple for get_llm_service."""
    path: str
    lora: Optional[str] = None
    lora_scale: float = 1.0

    @classmethod
    def from_any(cls, value: Any) -> Optional["ModelSpec"]:
        if value is None:
            return None
        if isinstance(value, str):
            return cls(path=value)
        if isinstance(value, dict):
            return cls(
                path=value["path"],
                lora=value.get("lora"),
                lora_scale=float(value.get("lora_scale", 1.0)),
            )
        raise TypeError(f"Unsupported model spec: {value!r}")
