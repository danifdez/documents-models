"""Type definitions for the agent system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class StepOutcome(Enum):
    CONTINUE = "continue"   # Step done, job re-enqueued as 'pending'
    FINISHED = "finished"   # Agent emitted finish, job marked 'processed'
    WAITING = "waiting"     # Agent dispatched a sub-agent, job is 'waiting'


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


@dataclass
class AgentDefinition:
    name: str
    system_prompt: str
    tools: List[str]
    model: Optional[ModelSpec] = None
    max_steps: int = 8
    max_depth: int = 2
    tool_defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    job_id: int
    job_type: str
    payload: Dict[str, Any]
    agent_def: AgentDefinition
    state: Dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: Dict[str, str]
    run: Callable[[Dict[str, Any], ToolContext], Dict[str, Any]]
    kind: str = "python"  # "python" | "task" | "sub_agent"
