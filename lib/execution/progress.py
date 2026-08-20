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
    max_tool_calls: int
    grant_id: Optional[str] = None

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
        max_tool_calls: int = 0,
    ) -> "ProgressLoopContext":
        loop_id = (
            str(emitter.context.get("executionId"))
            if emitter and emitter.context and loop_kind == "top_level"
            else str(uuid.uuid4())
        )
        requested_policy = {
            "normal": max_rounds,
            "repair": max_output_repairs,
            "closing": 1 if forced_finalization_available else 0,
            "maxTokensPerInference": max_tokens_per_inference,
            "toolCalls": max_tool_calls,
        }
        grant = None
        if (
            emitter
            and emitter.context
            and loop_kind == "top_level"
            and hasattr(emitter, "request_progress_grant")
        ):
            grant = emitter.request_progress_grant({
                "executionId": emitter.context["executionId"],
                "turnId": emitter.context.get("turnId"),
                "loopId": loop_id,
                "agentName": agent_name,
                "loopKind": loop_kind,
                "executionAttemptId": emitter.context.get("attemptId"),
                "requestedPolicy": requested_policy,
            })
        effective = (grant or {}).get("effectivePolicy", requested_policy)
        context = cls(
            loop_id=loop_id,
            agent_name=agent_name,
            loop_kind=loop_kind,
            max_rounds=int(effective["normal"]),
            max_output_repairs=int(effective["repair"]),
            forced_finalization_available=int(effective["closing"]) > 0,
            max_tokens_per_inference=int(effective["maxTokensPerInference"]),
            max_tool_calls=int(effective.get("toolCalls", max_tool_calls)),
            grant_id=(grant or {}).get("grantId"),
        )
        if emitter:
            emitter.record_progress_policy(context.policy())
            if emitter.context and loop_kind == "top_level":
                emitter.flush_evidence()
        return context

    def policy(self) -> Dict[str, Any]:
        value = {
            "version": "1",
            "source": (
                "documents.backend.progress_profile"
                if self.grant_id else "models.task_config"
            ),
            "loopId": self.loop_id,
            "agentName": self.agent_name,
            "loopKind": self.loop_kind,
            "maxRounds": self.max_rounds,
            "maxOutputRepairs": self.max_output_repairs,
            "forcedFinalizationAvailable": self.forced_finalization_available,
            "maxTokensPerInference": self.max_tokens_per_inference,
            "maxToolCalls": self.max_tool_calls,
        }
        if self.grant_id:
            value["grantId"] = self.grant_id
        return value

    def trace(
        self,
        *,
        round: int,
        phase: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        value = {
            **(extra or {}),
            "loopId": self.loop_id,
            "agentName": self.agent_name,
            "loopKind": self.loop_kind,
            "maxRounds": self.max_rounds,
            "round": round,
            "phase": phase,
        }
        if self.grant_id:
            value["budgetGrantId"] = self.grant_id
        return value
