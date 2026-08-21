"""Canonical semantic protocol shared by every agent runner."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class ModelOutcomeKind(str, Enum):
    FINAL_TEXT = "final_text"
    TOOL_REQUESTS = "tool_requests"
    STRUCTURED_RESULT = "structured_result"
    INVALID = "invalid"


class LoopStepOutcome(str, Enum):
    CONTINUE = "continue"
    FINISHED = "finished"
    WAITING = "waiting"


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: Optional[Dict[str, Any]]
    raw_arguments: str
    call_id: str = ""

    @classmethod
    def from_chat_call(cls, call: Dict[str, Any]) -> "ToolRequest":
        function = call.get("function") or {}
        raw = function.get("arguments")
        raw_arguments = (
            raw if isinstance(raw, str)
            else json.dumps(raw or {}, ensure_ascii=False)
        )
        try:
            parsed = json.loads(raw_arguments or "{}")
            arguments = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            arguments = None
        return cls(
            name=str(function.get("name") or ""),
            arguments=arguments,
            raw_arguments=raw_arguments,
            call_id=str(call.get("id") or ""),
        )

    @classmethod
    def from_structured(
        cls,
        name: Any,
        arguments: Any,
    ) -> Optional["ToolRequest"]:
        if not isinstance(arguments, dict):
            return None
        return cls(
            name=str(name or ""),
            arguments=arguments,
            raw_arguments=json.dumps(arguments, ensure_ascii=False),
        )

    def as_chat_call(self) -> Dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments,
            },
        }

    def same_operation(self, name: Any, arguments: Any) -> bool:
        return (
            self.name == str(name or "")
            and self.arguments is not None
            and isinstance(arguments, dict)
            and self.arguments == arguments
        )


@dataclass(frozen=True)
class ModelOutcome:
    kind: ModelOutcomeKind
    content: Optional[str] = None
    value: Optional[Dict[str, Any]] = None
    tool_requests: tuple[ToolRequest, ...] = ()
    reason: Optional[str] = None
    thought: Optional[str] = None

    @classmethod
    def from_chat_message(
        cls,
        content: str,
        tool_calls: List[Dict[str, Any]],
    ) -> "ModelOutcome":
        requests = tuple(ToolRequest.from_chat_call(call) for call in tool_calls)
        if requests:
            return cls(
                kind=ModelOutcomeKind.TOOL_REQUESTS,
                content=content,
                tool_requests=requests,
            )
        if content:
            return cls(kind=ModelOutcomeKind.FINAL_TEXT, content=content)
        return cls(
            kind=ModelOutcomeKind.INVALID,
            reason="empty_model_response",
        )

    @classmethod
    def from_structured_decision(cls, decision: Any) -> "ModelOutcome":
        if not isinstance(decision, dict):
            return cls(
                kind=ModelOutcomeKind.INVALID,
                reason="invalid_model_decision",
            )
        thought = (
            str(decision["thought"])
            if decision.get("thought") is not None
            else None
        )
        if "finish" in decision:
            finish = decision["finish"]
            value = finish if isinstance(finish, dict) else {"value": finish}
            return cls(
                kind=ModelOutcomeKind.STRUCTURED_RESULT,
                value=value,
                thought=thought,
            )
        if "tool" in decision:
            arguments = decision["args"] if "args" in decision else {}
            request = ToolRequest.from_structured(
                decision.get("tool"),
                arguments,
            )
            if request is None:
                return cls(
                    kind=ModelOutcomeKind.INVALID,
                    reason="invalid_tool_arguments",
                    thought=thought,
                )
            return cls(
                kind=ModelOutcomeKind.TOOL_REQUESTS,
                tool_requests=(request,),
                thought=thought,
            )
        return cls(
            kind=ModelOutcomeKind.INVALID,
            reason="invalid_model_decision",
            thought=thought,
        )


@dataclass(frozen=True)
class AgentRunResult:
    kind: Literal["final_text", "structured_result", "invalid"]
    content: Optional[str] = None
    value: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    completion_kind: Optional[Literal["full", "partial"]] = None
    completion_reason: Optional[str] = None
    completion_source: Optional[Literal["model", "runtime_template"]] = None
    partial_result: Optional[Dict[str, Any]] = None

    @classmethod
    def final_text(cls, content: str) -> "AgentRunResult":
        return cls(kind="final_text", content=content)

    @classmethod
    def partial_text(cls, content: str, reason: str) -> "AgentRunResult":
        return cls(
            kind="final_text",
            content=content,
            completion_kind="partial",
            completion_reason=reason,
            completion_source="model",
        )

    @classmethod
    def deterministic_partial_text(
        cls,
        content: str,
        reason: str,
        partial_result: Dict[str, Any],
    ) -> "AgentRunResult":
        return cls(
            kind="final_text",
            content=content,
            completion_kind="partial",
            completion_reason=reason,
            completion_source="runtime_template",
            partial_result=partial_result,
        )

    @classmethod
    def structured_result(cls, value: Dict[str, Any]) -> "AgentRunResult":
        return cls(kind="structured_result", value=value)

    @classmethod
    def invalid(cls, reason: str) -> "AgentRunResult":
        return cls(kind="invalid", reason=reason)

    def as_payload(self, *, text_field: str = "reply") -> Dict[str, Any]:
        if self.kind == "structured_result":
            payload = dict(self.value or {})
        elif self.kind == "final_text":
            payload = {text_field: self.content or ""}
        else:
            payload = {"error": self.reason or "invalid_agent_result"}
        if self.completion_kind:
            payload["completionKind"] = self.completion_kind
        if self.completion_reason:
            payload["completionReason"] = self.completion_reason
        if self.completion_source:
            payload["completionSource"] = self.completion_source
        if self.partial_result:
            payload["partialResult"] = self.partial_result
        return payload
