import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProgressLoopContext:
    loop_id: str
    agent_name: str
    loop_kind: str
    max_rounds: int
    normal_inference_soft_limit: int
    max_output_repairs: int
    forced_finalization_available: bool
    max_tokens_per_inference: int
    max_tool_calls: int
    tool_call_soft_limit: int
    exact_tool_repeat_warning: bool
    exact_tool_repeat_block_after_warning: bool
    grant_id: Optional[str] = None
    tool_budget_granted: int = 0
    tool_budget_available: int = 0
    tool_soft_limit_reached: bool = False
    soft_limit_warning_pending: bool = False
    soft_limit_warning_emitted: bool = False

    @classmethod
    def start(
        cls,
        emitter,
        *,
        agent_name: str,
        loop_kind: str,
        max_rounds: int,
        normal_inference_soft_limit: int = 2,
        max_output_repairs: int,
        forced_finalization_available: bool,
        max_tokens_per_inference: int,
        max_tool_calls: int = 0,
        tool_call_soft_limit: int = 0,
        exact_tool_repeat_warning: bool = False,
        exact_tool_repeat_block_after_warning: bool = False,
    ) -> "ProgressLoopContext":
        loop_id = (
            str(emitter.context.get("executionId"))
            if emitter and emitter.context and loop_kind == "top_level"
            else str(uuid.uuid4())
        )
        requested_policy = {
            "normal": max_rounds,
            "normalInferenceSoftLimit": normal_inference_soft_limit,
            "repair": max_output_repairs,
            "closing": 1 if forced_finalization_available else 0,
            "maxTokensPerInference": max_tokens_per_inference,
            "toolCalls": max_tool_calls,
            "toolCallSoftLimit": tool_call_soft_limit,
            "exactToolRepeatWarning": exact_tool_repeat_warning,
            "exactToolRepeatBlockAfterWarning": exact_tool_repeat_block_after_warning,
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
        historical_soft_limit = 0 if grant else normal_inference_soft_limit
        historical_tool_soft_limit = 0 if grant else tool_call_soft_limit
        historical_repeat_warning = False if grant else exact_tool_repeat_warning
        historical_repeat_block = (
            False if grant else exact_tool_repeat_block_after_warning
        )
        budget_state = (grant or {}).get("_budgetState") or {}
        tool_state = budget_state.get("tool") if isinstance(budget_state, dict) else {}
        tool_state = tool_state if isinstance(tool_state, dict) else {}
        soft_limit_reached = bool(tool_state.get("softLimitReached", False))
        context = cls(
            loop_id=loop_id,
            agent_name=agent_name,
            loop_kind=loop_kind,
            max_rounds=int(effective["normal"]),
            normal_inference_soft_limit=int(
                effective.get(
                    "normalInferenceSoftLimit",
                    historical_soft_limit,
                )
            ),
            max_output_repairs=int(effective["repair"]),
            forced_finalization_available=int(effective["closing"]) > 0,
            max_tokens_per_inference=int(effective["maxTokensPerInference"]),
            max_tool_calls=int(effective.get("toolCalls", max_tool_calls)),
            tool_call_soft_limit=int(
                effective.get("toolCallSoftLimit", historical_tool_soft_limit)
            ),
            exact_tool_repeat_warning=bool(
                effective.get(
                    "exactToolRepeatWarning",
                    historical_repeat_warning,
                )
            ),
            exact_tool_repeat_block_after_warning=bool(
                effective.get(
                    "exactToolRepeatBlockAfterWarning",
                    historical_repeat_block,
                )
            ),
            grant_id=(grant or {}).get("grantId"),
            tool_budget_granted=int(tool_state.get("granted", 0)),
            tool_budget_available=int(tool_state.get("available", 0)),
            tool_soft_limit_reached=soft_limit_reached,
            soft_limit_warning_pending=soft_limit_reached,
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
            "normalInferenceSoftLimit": self.normal_inference_soft_limit,
            "maxOutputRepairs": self.max_output_repairs,
            "forcedFinalizationAvailable": self.forced_finalization_available,
            "maxTokensPerInference": self.max_tokens_per_inference,
            "maxToolCalls": self.max_tool_calls,
            "toolCallSoftLimit": self.tool_call_soft_limit,
            "exactToolRepeatWarning": self.exact_tool_repeat_warning,
            "exactToolRepeatBlockAfterWarning": (
                self.exact_tool_repeat_block_after_warning
            ),
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

    def observe_tool_budget(self, handle) -> None:
        state = getattr(handle, "budget_state", None)
        tool = state.get("tool") if isinstance(state, dict) else None
        if not isinstance(tool, dict):
            return
        self.tool_budget_granted = int(tool.get("granted", self.tool_budget_granted))
        self.tool_budget_available = int(
            tool.get("available", self.tool_budget_available)
        )
        reached = bool(tool.get("softLimitReached", False))
        self.tool_soft_limit_reached = reached
        if reached and not self.soft_limit_warning_emitted:
            self.soft_limit_warning_pending = True

    def messages_for_inference(self, messages):
        if not self.soft_limit_warning_pending or self.soft_limit_warning_emitted:
            return messages
        self.soft_limit_warning_pending = False
        self.soft_limit_warning_emitted = True
        warning = (
            f"Tool budget is low: {self.tool_budget_available} of "
            f"{self.tool_budget_granted} calls remain. Prefer completing with "
            "the evidence already collected. Use another tool only when it is "
            "required to answer the user correctly; do not open optional work."
        )
        return [*messages, {"role": "system", "content": warning}]
