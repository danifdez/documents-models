"""Agent abstraction: an LLM tool-calling loop with a declared tool set.

An `AgentSpec` describes an agent — its prompt, the tools it may use (by name,
resolved against the shared `core/tools` repository), how many tool rounds it
gets, and how it finishes (a free-text reply, or a structured object matching an
output schema). The personal assistant and every subagent are instances of the
same abstraction; the only difference is the fields they set.

`run_agent_loop` is the single tool-round loop, shared by all agents. It is pure
mechanism: it receives the resolved tool schemas and a `dispatch` callable, so it
knows nothing about which tools are leaves and which are nested agents.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from services.model_config import get_llm_params, get_task_config
from services.llm_service import get_llm_service
from common.chat.http import post_tool_event
from common.chat.text_utils import strip_thinking as _strip_thinking
from tools import REGISTRY, summarize_leaf

logger = logging.getLogger(__name__)

# dispatch: (name, args_json, ctx) -> result dict. Resolves a tool call to a
# leaf executor or a nested agent run. Injected so this module stays agnostic.
DispatchFn = Callable[[str, str, Any], Dict[str, Any]]


@dataclass(frozen=True)
class AgentSpec:
    """An agent: prompt + declared tools + turn budget + how it finishes.

    - `tool_names`: the tools this agent may call, by name. Leaves live in
      `core/tools`; a name that resolves to another agent makes that agent
      callable-as-a-tool from here.
    - `output_schema`: None → the agent ends with a free-text reply (the caller
      decides what to do with the conversation). A dict → the agent must end
      with a JSON object of that shape; `run_agent_loop` returns it.
    - `emits_tool_events`: push live tool cards to the UI (the user-facing
      assistant does; subagents don't).
    - `tool_schema`: how this agent is offered to a PARENT agent that lists it in
      its own `tool_names`. None for a top-level agent that nobody invokes.
    - `input_field`: when invoked as a tool, the JSON property holding its input.
    - `requires_folder`: this agent is useless without a working folder, so a
      parent hides it from its catalog when the turn carries no `folderScope`
      (offering it anyway just degrades tool selection).
    """
    name: str
    config_key: str
    system_prompt: str
    tool_names: frozenset
    max_rounds: int = 3
    output_schema: Optional[Dict[str, Any]] = None
    emits_tool_events: bool = False
    tool_schema: Optional[Dict[str, Any]] = None
    input_field: str = "query"
    fallback_max_tokens: int = 600
    requires_folder: bool = False


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
    spec: AgentSpec, llm, messages: List[Dict[str, Any]], content: str, max_tokens: int,
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


def run_agent_loop(
    spec: AgentSpec,
    messages: List[Dict[str, Any]],
    ctx,
    tools: List[Dict[str, Any]],
    dispatch: DispatchFn,
) -> Optional[Dict[str, Any]]:
    """Run tool-call rounds until the model replies without calling a tool or the
    round budget runs out. Mutates `messages` in place (appends assistant/tool
    turns). Returns the structured object for a schema agent, or None for a
    free-reply agent (the caller renders the reply from the augmented messages)."""
    cfg = get_task_config(spec.config_key)
    params = get_llm_params(spec.config_key)
    llm = get_llm_service(**params)
    max_rounds = int(cfg.get("max_rounds", spec.max_rounds))
    round_max_tokens = int(
        cfg.get("tool_max_tokens") or cfg.get("max_tokens") or spec.fallback_max_tokens
    )
    can_emit = (
        spec.emits_tool_events
        and isinstance(ctx.owner_id, int)
        and isinstance(ctx.job_id, int)
    )

    logger.info(
        "agent[%s]: enter — tools=%s rounds=%d", spec.name,
        sorted(spec.tool_names), max_rounds,
    )

    for round_idx in range(max_rounds):
        msg = llm.chat_with_tools(messages, tools, max_tokens=round_max_tokens)
        content = _strip_thinking(msg.get("content") or "")
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls and content:
            tool_calls = extract_inline_tool_calls(content)
        logger.info(
            "agent[%s]: round %d → tool_calls=%d content_head=%r",
            spec.name, round_idx, len(tool_calls), content[:120],
        )
        if not tool_calls:
            if spec.output_schema is not None:
                return _coerce_output(spec, llm, messages, content, round_max_tokens)
            return None

        messages.append({
            "role": "assistant",
            "content": msg.get("content") or None,
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args_json = fn.get("arguments") or "{}"
            try:
                args_obj = json.loads(args_json or "{}")
                args_label = str(args_obj.get("query") or next(iter(args_obj.values()), ""))
            except (json.JSONDecodeError, StopIteration):
                args_label = ""
            if can_emit:
                post_tool_event(
                    ctx.owner_segment, ctx.owner_id, ctx.job_id, name, args_label,
                    status="running",
                )
            result = dispatch(name, args_json, ctx)
            summary, entity = _summarize(name, result)
            logger.info("agent[%s]: tool=%s → %s", spec.name, name, summary)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
            pending = isinstance(result, dict) and result.get("pendingConfirmation")
            if can_emit and not pending:
                post_tool_event(
                    ctx.owner_segment, ctx.owner_id, ctx.job_id, name, args_label,
                    status="done", summary=summary, entity=entity,
                )
            # A schema agent invoked as a tool must yield control the moment an
            # inner tool parks a confirmation card — the parent responds to it.
            if pending and spec.output_schema is not None:
                logger.info("agent[%s]: pending confirmation from %s, ending",
                            spec.name, name)
                return {
                    "summary": (
                        f"Awaiting user confirmation for {name}. A confirmation "
                        "card has been shown to the user."
                    ),
                }

    # Out of rounds. A schema agent still owes a structured object.
    logger.info("agent[%s]: rounds exhausted", spec.name)
    if spec.output_schema is not None:
        return _coerce_output(spec, llm, messages, "", round_max_tokens)
    return None
