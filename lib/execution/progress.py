import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ProgressLoopContext:
    loop_id: str
    agent_name: str
    loop_kind: str
    max_rounds: int
    max_output_repairs: int
    forced_finalization_available: bool
    max_tokens_per_inference: int

    @classmethod
    def start(
        cls,
        emitter,
        *,
        agent_name: str,
        loop_kind: str,
        max_rounds: int,
        max_output_repairs: int,
        forced_finalization_available: bool,
        max_tokens_per_inference: int,
    ) -> "ProgressLoopContext":
        context = cls(
            loop_id=str(uuid.uuid4()),
            agent_name=agent_name,
            loop_kind=loop_kind,
            max_rounds=max_rounds,
            max_output_repairs=max_output_repairs,
            forced_finalization_available=forced_finalization_available,
            max_tokens_per_inference=max_tokens_per_inference,
        )
        if emitter:
            emitter.record_progress_policy(context.policy())
        return context

    def policy(self) -> Dict[str, Any]:
        return {
            "version": "1",
            "source": "models.task_config",
            "loopId": self.loop_id,
            "agentName": self.agent_name,
            "loopKind": self.loop_kind,
            "maxRounds": self.max_rounds,
            "maxOutputRepairs": self.max_output_repairs,
            "forcedFinalizationAvailable": self.forced_finalization_available,
            "maxTokensPerInference": self.max_tokens_per_inference,
        }

    def trace(
        self,
        *,
        round: int,
        phase: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            **(extra or {}),
            "loopId": self.loop_id,
            "agentName": self.agent_name,
            "loopKind": self.loop_kind,
            "maxRounds": self.max_rounds,
            "round": round,
            "phase": phase,
        }
