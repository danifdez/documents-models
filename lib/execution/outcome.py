from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class InferenceOutcome:
    value: Dict[str, Any]
