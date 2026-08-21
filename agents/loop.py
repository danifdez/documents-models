"""The agent engine: the single tool-round loop shared by every agent.

`run_agent_loop` is pure mechanism. It receives the resolved tool schemas and a
`dispatch` callable, so it knows nothing about which tools are leaves and which
are nested agents. It reads an `AgentSpec` (the abstraction, from
`lib.framework.agent`) to know how the agent finishes, and leans on the shared
tool repository (`core/tools`) only to summarise tool results for the UI card.

The harness patches `extract_inline_tool_calls` in this module to count how many
`<tool_call>` blocks the model emitted vs how many parsed.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from lib.framework.agent import AgentRunResult, AgentSpec
from lib.execution import (
    InferenceBudgetDenied,
    ProgressLoopContext,
    ToolBudgetDenied,
    sanitize_execution_value,
    sanitize_result_summary,
)
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.text import strip_thinking as _strip_thinking
from lib.backend.http import post_tool_event
from services.llm_service import get_llm_service
from tools import REGISTRY, summarize_leaf

logger = logging.getLogger(__name__)

# dispatch: (name, args_json, ctx) -> result dict. Resolves a tool call to a
# leaf executor or a nested agent run. Injected so this module stays agnostic.
DispatchFn = Callable[[str, str, Any], Dict[str, Any]]


# ── Inline tool-call parsing (Qwen3 emits <tool_call>…</tool_call> in content) ─
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_inline_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Fallback parser: llama-cpp doesn't always lift inline <tool_call> blocks
    into the OpenAI-shaped `tool_calls` field, so we recover them from content."""
    if not content or "<tool_call>" not in content:
        return []
    out: List[Dict[str, Any]] = []
    for i, match in enumerate(_TOOL_CALL_RE.finditer(content)):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = str(obj.get("name") or "").strip()
        if not name:
            continue
        arguments = obj.get("arguments")
        args_str = (
            arguments if isinstance(arguments, str)
            else json.dumps(arguments or {}, ensure_ascii=False)
        )
        out.append({
            "id": f"inline_call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    return out


_FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

_OUTPUT_REPAIR_PROMPT = (
    "The previous model output was empty and invalid. Continue from the current "
    "conversation and tool results. If another tool is needed, call it. "
    "Otherwise return the final answer. Do not repeat a completed tool unless "
    "its result requires a different request."
)

_FORCED_FINALIZATION_PROMPT = (
    "The operation budget is exhausted. Reply now with the best final answer "
    "supported by the evidence already available. Do not call tools and do not "
    "emit tool-call markup."
)

_DETERMINISTIC_PARTIAL_LIMIT = 5
_CLOSING_UNAVAILABLE_REASONS = {
    "budget_hard_limit_reached",
    "budget_reservation_consumed",
}


def _json_object_or_none(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    m = _FENCED_JSON_RE.match(t)
    if m:
        t = m.group(1)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _coerce_output(
    spec: AgentSpec,
    llm,
    messages: List[Dict[str, Any]],
    content: str,
    max_tokens: int,
    trace_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Turn the agent's final text into the JSON object its output schema
    promises. Fast path: the reply is usually already that object (the schema is
    in its system prompt). Otherwise regenerate once with the schema as
    `response_format` so decoding is constrained to its shape."""
    if spec.output_schema is None:
        return {"summary": content}
    required = spec.output_schema.get("required") or []
    parsed = _json_object_or_none(content)
    if parsed is not None and all(k in parsed for k in required):
        return parsed
    logger.info("agent[%s]: reply not schema-shaped, constraining decode", spec.name)
    try:
        followup = list(messages)
        if content:
            followup.append({"role": "assistant", "content": content})
        followup.append({
            "role": "user",
            "content": (
                "Reply now with ONLY the JSON object matching the OUTPUT SCHEMA "
                "in your instructions. No other text."
            ),
        })
        forced = llm.chat(
            followup, max_tokens=max_tokens,
            response_format={"type": "json_object", "schema": spec.output_schema},
            inference_name="structured_output_repair",
            trace_metadata={
                **(trace_metadata or {}),
                "phase": "structured_output_repair",
            },
        ) or ""
        parsed = _json_object_or_none(_strip_thinking(forced))
        if parsed is not None:
            return parsed
    except Exception:
        logger.exception("agent[%s]: schema-constrained retry failed", spec.name)
    return {"summary": content or ""}


def _summarize(name: str, result: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """One-line tool-card summary. A nested agent returns a {summary,…} object
    (its name is not a leaf in the registry); leaves summarise themselves."""
    if (isinstance(result, dict) and name not in REGISTRY
            and isinstance(result.get("summary"), str)):
        text = result["summary"]
        return text[:200] + ("…" if len(text) > 200 else ""), None
    return summarize_leaf(name, result)


def _normalize_message(msg: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    content = _strip_thinking(msg.get("content") or "")
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls and content:
        tool_calls = extract_inline_tool_calls(content)
    return content, tool_calls


def _tool_label(name: str) -> str:
    labels = {
        "search_workspace": "Workspace search",
        "update_task": "Update task",
        "delete_task": "Delete task",
        "mark_event_occurrence_done": "Mark event done",
        "set_task_reminder": "Set task reminder",
        "clear_task_reminder": "Clear task reminder",
        "folder_search": "Folder search",
        "folder_read": "Folder read",
        "folder_write": "Folder write",
        "folder_delete": "Folder delete",
    }
    if name in labels:
        return labels[name]
    return re.sub(r"[_-]+", " ", name).strip().title()


def _deterministic_partial(
    progress: ProgressLoopContext,
    execution,
    completed_operations: List[Dict[str, str]],
    *,
    tool_budget_exhausted: bool,
    trigger: str,
) -> Optional[AgentRunResult]:
    context = getattr(execution, "context", None)
    execution_attempt_id = (
        str(context.get("attemptId"))
        if isinstance(context, dict) and context.get("attemptId")
        else ""
    )
    if (
        not completed_operations
        or progress.loop_kind != "top_level"
        or not progress.grant_id
        or not execution_attempt_id
    ):
        return None
    reason = (
        "tool_budget_exhausted" if tool_budget_exhausted else "budget_exhausted"
    )
    limit_label = "tool-call limit" if tool_budget_exhausted else "execution limit"
    visible = completed_operations[:_DETERMINISTIC_PARTIAL_LIMIT]
    lines = [
        f"I couldn't produce the final synthesis because this turn reached its {limit_label}.",
        "",
        "Completed work:",
        *[
            f"- {_tool_label(item['name'])}: {item['summary']}"
            for item in visible
        ],
    ]
    remaining = len(completed_operations) - len(visible)
    if remaining:
        lines.append(
            f"- {remaining} additional completed operation"
            f"{'s are' if remaining != 1 else ' is'} available in the activity log."
        )
    lines.extend([
        "",
        "Pending:",
        "- Final synthesis of the completed results.",
    ])
    partial_result = {
        "version": "1",
        "trigger": trigger,
        "loopId": progress.loop_id,
        "grantId": progress.grant_id,
        "executionAttemptId": execution_attempt_id,
        "completedOperations": completed_operations,
        "pending": ["final_synthesis"],
    }
    return AgentRunResult.deterministic_partial_text(
        "\n".join(lines),
        reason,
        partial_result,
    )


def run_agent_loop(
    spec: AgentSpec,
    messages: List[Dict[str, Any]],
    ctx,
    tools: List[Dict[str, Any]],
    dispatch: DispatchFn,
    loop_kind: str = "top_level",
) -> AgentRunResult:
    """Run tool-call rounds until the model replies without calling a tool or the
    round budget runs out. Mutates `messages` in place (appends assistant/tool
    turns). Returns the final outcome produced by the loop."""
    cfg = get_task_config(spec.config_key)
    params = get_llm_params(spec.config_key)
    llm = get_llm_service(**params)
    max_rounds = int(cfg.get("max_rounds", spec.max_rounds))
    normal_inference_soft_limit = int(
        cfg.get("normal_inference_soft_limit", 2)
    )
    round_max_tokens = int(
        cfg.get("tool_max_tokens") or cfg.get("max_tokens") or spec.fallback_max_tokens
    )
    max_tool_calls = int(cfg.get("max_tool_calls", 6))
    tool_call_soft_limit = int(cfg.get("tool_call_soft_limit", 4))
    exact_tool_repeat_warning = bool(cfg.get("exact_tool_repeat_warning", True))
    can_emit = (
        spec.emits_tool_events
        and isinstance(ctx.owner_id, int)
        and isinstance(ctx.execution_id, str)
    )
    execution = getattr(ctx, "execution", None)
    output_repair_used = False
    progress = ProgressLoopContext.start(
        execution,
        agent_name=spec.name,
        loop_kind=loop_kind,
        max_rounds=max_rounds,
        normal_inference_soft_limit=normal_inference_soft_limit,
        max_output_repairs=2 if spec.output_schema is not None else 1,
        forced_finalization_available=spec.output_schema is None,
        max_tokens_per_inference=round_max_tokens,
        max_tool_calls=max_tool_calls,
        tool_call_soft_limit=tool_call_soft_limit,
        exact_tool_repeat_warning=exact_tool_repeat_warning,
    )
    max_rounds = progress.max_rounds
    round_max_tokens = progress.max_tokens_per_inference

    logger.info(
        "agent[%s]: enter — tools=%s rounds=%d", spec.name,
        sorted(spec.tool_names), max_rounds,
    )

    closing_reason = "step_budget_exhausted"
    closing_round = max_rounds
    tool_budget_exhausted = False
    completed_operations: List[Dict[str, str]] = []
    pending_confirmation = False
    for round_idx in range(max_rounds):
        round_trace = progress.trace(
            round=round_idx + 1,
            phase="agent_loop",
        )
        try:
            msg = llm.chat_with_tools(
                progress.messages_for_inference(messages),
                tools,
                max_tokens=round_max_tokens,
                inference_name="chat_with_tools",
                trace_metadata=round_trace,
            )
        except InferenceBudgetDenied as error:
            return AgentRunResult.invalid(error.reason)
        operation_phase = "agent_loop"
        content, tool_calls = _normalize_message(msg)
        logger.info(
            "agent[%s]: round %d → tool_calls=%d content_head=%r",
            spec.name, round_idx, len(tool_calls), content[:120],
        )
        if (
            not tool_calls
            and not content
            and spec.output_schema is None
            and tools
            and not output_repair_used
            and progress.max_output_repairs > 0
        ):
            output_repair_used = True
            repair_messages = list(messages)
            repair_messages.append({
                "role": "system",
                "content": _OUTPUT_REPAIR_PROMPT,
            })
            try:
                msg = llm.chat_with_tools(
                    repair_messages,
                    tools,
                    max_tokens=round_max_tokens,
                    inference_name="output_repair",
                    trace_metadata=progress.trace(
                        round=round_idx + 1,
                        phase="output_repair",
                        extra={
                            "reason": "empty_model_response",
                            "attempt": 1,
                            "maxAttempts": 1,
                        },
                    ),
                )
            except InferenceBudgetDenied as error:
                return AgentRunResult.invalid(error.reason)
            operation_phase = "output_repair"
            content, tool_calls = _normalize_message(msg)
            logger.info(
                "agent[%s]: output repair → tool_calls=%d content_head=%r",
                spec.name, len(tool_calls), content[:120],
            )
        if not tool_calls:
            if spec.output_schema is not None:
                return AgentRunResult.structured_result(
                    _coerce_output(
                        spec,
                        llm,
                        messages,
                        content,
                        round_max_tokens,
                        round_trace,
                    )
                )
            if content:
                return AgentRunResult.final_text(content)
            reason = (
                "empty_model_response_after_repair"
                if output_repair_used
                else "empty_model_response"
            )
            return AgentRunResult.invalid(reason)

        messages.append({
            "role": "assistant",
            "content": msg.get("content") or None,
            "tool_calls": tool_calls,
        })
        for call_index, call in enumerate(tool_calls):
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args_json = fn.get("arguments") or "{}"
            arguments_are_object = False
            try:
                parsed_args = json.loads(args_json or "{}")
                if not isinstance(parsed_args, dict):
                    raise TypeError("Tool arguments must be an object")
                args_obj = parsed_args
                arguments_are_object = True
                args_label = str(args_obj.get("query") or next(iter(args_obj.values()), ""))
            except (json.JSONDecodeError, StopIteration, TypeError):
                args_obj = {}
                args_label = ""
            tool_trace = None
            if execution:
                try:
                    tool_trace = execution.start_tool(
                        name,
                        args_obj,
                        str(call.get("id") or "") or None,
                        progress.trace(
                            round=round_idx + 1,
                            phase=operation_phase,
                        ),
                        repeat_comparable=(
                            progress.exact_tool_repeat_warning
                            and name in REGISTRY
                            and arguments_are_object
                        ),
                    )
                except ToolBudgetDenied:
                    for skipped_call in tool_calls[call_index:]:
                        skipped_fn = skipped_call.get("function") or {}
                        messages.append({
                            "role": "tool",
                            "tool_call_id": skipped_call.get("id") or "",
                            "name": str(skipped_fn.get("name") or ""),
                            "content": json.dumps({
                                "error": "tool_budget_hard_limit_reached",
                                "skipped": True,
                            }),
                        })
                    tool_budget_exhausted = True
                    closing_reason = "tool_budget_exhausted"
                    closing_round = round_idx + 1
                    break
                execution.flush_evidence()
            if can_emit:
                post_tool_event(
                    ctx.owner_segment, ctx.owner_id, ctx.execution_id, name, args_label,
                    status="running",
                )
            try:
                result = dispatch(name, args_json, ctx)
            except Exception as error:
                if execution and tool_trace:
                    execution.finish_tool(
                        tool_trace,
                        {},
                        None,
                        error=str(error),
                    )
                    execution.flush_evidence()
                raise
            source_event_id = None
            summary, entity = _summarize(name, result)
            visible_summary = sanitize_result_summary(summary)
            is_leaf_tool = name in REGISTRY
            tool_error = (
                str(result.get("error"))
                if isinstance(result, dict) and result.get("error")
                else None
            )
            if execution and tool_trace:
                source_event_id = execution.observe_tool_result(tool_trace, result)
                execution.finish_tool(
                    tool_trace,
                    result,
                    source_event_id,
                    error=tool_error,
                    result_summary=(visible_summary if is_leaf_tool and not tool_error else None),
                    result_summary_kind=("leaf_tool" if is_leaf_tool and not tool_error else None),
                )
                execution.flush_evidence()
                progress.observe_tool_budget(tool_trace)
                if (
                    is_leaf_tool
                    and visible_summary
                    and not tool_error
                ):
                    completed_operations.append({
                        "operationId": tool_trace.operation_id,
                        "toolCallId": tool_trace.tool_call_id or "",
                        "name": name,
                        "summary": visible_summary,
                    })
            logger.info(
                "agent[%s]: tool=%s → %s", spec.name, name, visible_summary
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "name": name,
                "content": json.dumps(
                    sanitize_execution_value(result),
                    ensure_ascii=False,
                ),
            })
            pending = isinstance(result, dict) and result.get("pendingConfirmation")
            pending_confirmation = pending_confirmation or bool(pending)
            if can_emit and not pending:
                post_tool_event(
                    ctx.owner_segment, ctx.owner_id, ctx.execution_id, name, args_label,
                    status="done", summary=visible_summary, entity=entity,
                )
            # A schema agent invoked as a tool must yield control the moment an
            # inner tool parks a confirmation card — the parent responds to it.
            if pending and spec.output_schema is not None:
                logger.info("agent[%s]: pending confirmation from %s, ending",
                            spec.name, name)
                return AgentRunResult.structured_result({
                    "summary": (
                        f"Awaiting user confirmation for {name}. A confirmation "
                        "card has been shown to the user."
                    ),
                })
        if tool_budget_exhausted:
            break

    logger.info("agent[%s]: closing after %s", spec.name, closing_reason)
    if spec.output_schema is not None:
        return AgentRunResult.structured_result(
            _coerce_output(
                spec,
                llm,
                messages,
                "",
                round_max_tokens,
                progress.trace(
                    round=closing_round,
                    phase="structured_output_repair",
                ),
            )
        )
    if not progress.forced_finalization_available:
        if not pending_confirmation:
            partial = _deterministic_partial(
                progress,
                execution,
                completed_operations,
                tool_budget_exhausted=tool_budget_exhausted,
                trigger="closing_unavailable",
            )
            if partial:
                return partial
        return AgentRunResult.invalid(
            "tool_budget_exhausted_without_closing"
            if tool_budget_exhausted else "budget_hard_limit_reached"
        )
    try:
        closing_messages = [
            *messages,
            {"role": "system", "content": _FORCED_FINALIZATION_PROMPT},
        ]
        forced = llm.chat(
            closing_messages,
            max_tokens=round_max_tokens,
            allow_thinking=True,
            inference_name="forced_finalization",
            trace_metadata=progress.trace(
                round=closing_round,
                phase="forced_finalization",
                extra={"reason": closing_reason},
            ),
        ) or ""
    except InferenceBudgetDenied as error:
        if error.reason in _CLOSING_UNAVAILABLE_REASONS and not pending_confirmation:
            partial = _deterministic_partial(
                progress,
                execution,
                completed_operations,
                tool_budget_exhausted=tool_budget_exhausted,
                trigger="closing_unavailable",
            )
            if partial:
                return partial
        if tool_budget_exhausted and error.reason in {
            "budget_hard_limit_reached",
            "budget_reservation_consumed",
        }:
            return AgentRunResult.invalid(
                "tool_budget_exhausted_without_closing"
            )
        return AgentRunResult.invalid(error.reason)
    content = _strip_thinking(forced)
    if content:
        return AgentRunResult.partial_text(
            content,
            "tool_budget_exhausted" if tool_budget_exhausted else "budget_exhausted",
        )
    if not pending_confirmation:
        partial = _deterministic_partial(
            progress,
            execution,
            completed_operations,
            tool_budget_exhausted=tool_budget_exhausted,
            trigger="closing_output_empty",
        )
        if partial:
            return partial
    return AgentRunResult.invalid(
        "tool_budget_empty_forced_finalization"
        if tool_budget_exhausted else "budget_empty_forced_finalization"
    )
