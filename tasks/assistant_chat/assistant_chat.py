"""Personal assistant and agent handler.

Multi-turn chat with system prompt and history. This is NOT a Q&A over a file
(that's `ask`). The thread is persisted on the NestJS backend.

Expected payload (common fields):
  {
    "kind": "assistant" | "agent",        # default "assistant" for back-compat
    "ownerType": "main-assistant" | "agent",
    "ownerId": int,                       # id in owner's table
    "name": str,                          # owner's display name
    "systemPrompt": str | null,           # owner's custom prompt; null => default
    "folderScope": str | null,
    "conversation": [{"role": ..., "content": ...}, ...]
  }

For kind="assistant" additionally:
  "assistantId": int,                     # legacy alias of ownerId
  "assistantName": str,                   # legacy alias of name
  "assistantSystem": bool,                # true for the personal assistant
  "memorySnippets": [...],                # injected memory
  "extractMemory": bool                   # run memory extraction

For kind="agent" additionally:
  "agentId": int                          # legacy alias of ownerId

Returns:
  {"reply": str, "memoryAction"?: {...}}  or  {"error": str}
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import urllib.error
import urllib.request

from services.llm_service import get_llm_service
from services.model_config import get_llm_params, get_task_config
from services.prompts import load_prompt
from utils.job_registry import job_handler

from tasks.assistant_chat.tools import (
    AGENT_ALLOWED_TOOLS,
    ASSISTANT_TOOLS,
    SUBAGENT_SPECS,
    SubagentSpec,
    ToolContext,
    execute_leaf,
    summarize as summarize_tool_result,
    tools_for_payload,
)
from common.chat.http import BACKEND_URL, post_tool_event
from common.chat.text_utils import strip_thinking as _strip_thinking
from tasks.assistant_chat.memory_agent import (
    extract_memory_action as _extract_memory_action,
    format_memory_block as _format_memory_block,
    last_user_message as _last_user_message,
    memory_for_payload as _memory_for_payload,
)

logger = logging.getLogger(__name__)

# How many rounds of tool calls we allow per user turn before forcing a final
# text response. Prevents the model from looping on tools forever.
MAX_TOOL_ROUNDS = 3

def _owner_id(payload: Dict[str, Any]) -> Optional[int]:
    """Resolve the owner id for streaming / tool-event POSTs.
    Prefers `ownerId`; falls back to legacy `assistantId` or `agentId`."""
    for key in ("ownerId", "assistantId", "agentId"):
        v = payload.get(key)
        if isinstance(v, int):
            return v
    return None


def _owner_type(payload: Dict[str, Any]) -> str:
    """'main-assistant' or 'agent'. Defaults to legacy assistant."""
    t = payload.get("ownerType")
    if t in ("main-assistant", "agent"):
        return t
    return "agent" if (payload.get("kind") == "agent") else "main-assistant"


def _backend_owner_segment(payload: Dict[str, Any]) -> str:
    """`/assistants/:id` or `/agents/:id` prefix for indexed-files endpoints."""
    return "agents" if _owner_type(payload) == "agent" else "assistants"


_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

DEFAULT_AGENT_SYSTEM_PROMPT = load_prompt(_PROMPTS_DIR, "agent_system.md").strip()
DEFAULT_ASSISTANT_SYSTEM_PROMPT = load_prompt(_PROMPTS_DIR, "assistant_system.md").strip()

# Multi-tool composition orientation, injected into the system prompt of the
# main assistant and of agents alike. Tool-agnostic: agents only see four
# folder_* leaf tools (no subagents), but the rules (compose, chain, don't
# repeat, stop when done) still apply. The reference to "subagent tools" is
# benign for agents since none appear in their filtered catalog.
MULTI_TOOL_ORIENTATION = load_prompt(_PROMPTS_DIR, "multi_tool_orientation.md").strip()


# Flush a partial chunk to the backend roughly every N ms, regardless of how
# many tokens have accumulated. Tuned so the user sees forward motion without
# drowning the HTTP loop in tiny requests.
STREAM_FLUSH_INTERVAL_MS = 120


def _post_stream_chunk(
    owner_segment: str,
    owner_id: int,
    job_id: int,
    chunk: str,
    done: bool = False,
) -> None:
    """Best-effort POST of a partial reply chunk back to the backend. Failures
    are logged but never raised — streaming is purely a UX nicety; the final
    reply still arrives through the normal job-result path.

    When `done=True`, signals that the model has finished generating even if
    the job hasn't fully completed yet (memory extraction may still be
    running). The UI uses this to stop the live caret immediately instead of
    waiting for the final assistantResponse event."""
    if not chunk and not done:
        return
    url = f"{BACKEND_URL}/{owner_segment}/{owner_id}/stream-chunk"
    body = {"jobId": job_id, "chunk": chunk}
    if done:
        body["done"] = True
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2):
            pass
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning("assistant-chat: stream-chunk POST failed: %s", e)

# How many turns of history we keep as context. The backend persists
# the full thread, but it makes no sense to pass it entirely to the LLM on every
# call — it falls out of context and latency spikes.
DEFAULT_HISTORY_TURNS = 16


def _build_messages(payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    kind = (payload.get("kind") or "assistant")
    raw_system_prompt = (payload.get("systemPrompt") or "").strip()
    # No custom prompt → use this service's default for the kind. The behaviour
    # prompt lives here, next to the tools — not in the backend.
    if not raw_system_prompt:
        system_prompt = (
            DEFAULT_AGENT_SYSTEM_PROMPT if kind == "agent"
            else DEFAULT_ASSISTANT_SYSTEM_PROMPT
        )
    else:
        system_prompt = raw_system_prompt
    folder_scope = (payload.get("folderScope") or "").strip()
    memory_snippets = _memory_for_payload(payload) if kind == "assistant" else []
    conversation = payload.get("conversation") or []

    # Multi-tool composition orientation. Concatenated into the same system
    # message (a second system message tends to be ignored), between the
    # assistant/agent persona and the persistent memory block. Applies to
    # both kinds: agents only see leaf folder_* tools, but the same rules on
    # composition, chaining and stopping still apply.
    system_prompt = (
        f"{system_prompt}\n\n{MULTI_TOOL_ORIENTATION}"
        if system_prompt
        else MULTI_TOOL_ORIENTATION
    )

    if kind == "assistant":
        now = datetime.now().astimezone().replace(microsecond=0)
        system_prompt = (
            f"{system_prompt}\n\nCurrent date and time: {now.isoformat()} "
            f"({now:%A}). Resolve relative or time-only references like "
            "'at 10pm', 'tomorrow' or 'next Monday' against it when scheduling."
        )

    # Persistent user memory only applies to the personal assistant. Agents
    # don't have memory; skip the injection entirely (and the prompt section
    # below) when kind != 'assistant'.
    memory_block = _format_memory_block(memory_snippets)
    if memory_block:
        memory_section = (
            "What you already know about the user (persistent memory from "
            "previous conversations). Lean on these facts when they are "
            "relevant to your answer; if the user asks something whose "
            "answer is here, use it directly:\n"
            f"{memory_block}"
        )
        system_prompt = (
            f"{system_prompt}\n\n{memory_section}" if system_prompt else memory_section
        )

    # Qwen3 emits <think>...</think> before every response by default.
    # That consumes tokens (high latency). For conversational chat it adds
    # no value; we disable it via the official `/no_think` flag in the system
    # prompt. Configurable in case we want explicit thinking in another phase.
    enable_thinking = bool(cfg.get("enable_thinking", False))
    if not enable_thinking:
        marker = "/no_think"
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{marker}"
        else:
            system_prompt = marker

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # If the helper has an associated folder, we keep it as a separate
    # context block. No tools use it yet, but the model can
    # mention it when asked.
    if folder_scope:
        messages.append({
            "role": "system",
            "content": f"[WORKING FOLDER]\nPath: {folder_scope}",
        })

    history_turns = int(cfg.get("history_turns", DEFAULT_HISTORY_TURNS))
    # Filter to valid roles and keep the last N turns.
    filtered = [
        {"role": m["role"], "content": str(m.get("content") or "")}
        for m in conversation
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and m.get("content")
    ]
    messages.extend(filtered[-history_turns:])

    return messages


# Qwen3 emits tool calls inline as <tool_call>{"name":..., "arguments":{...}}</tool_call>.
# llama-cpp-python doesn't always parse these into the OpenAI-compatible
# `tool_calls` field, so we extract them ourselves from the message content
# as a fallback. Each match yields a synthetic tool_call dict with the same
# shape the official path would have produced.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_inline_tool_calls(content: str) -> List[Dict[str, Any]]:
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
        # Tool API expects `arguments` as a JSON string, not a dict.
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


def _execute_tool(name: str, args_json: str, ctx: ToolContext) -> Dict[str, Any]:
    """Dispatch a single tool call. Returns the result as a dict (will be
    JSON-encoded by the caller before feeding back to the model).

    Defensive: for agents, reject any tool name outside AGENT_ALLOWED_TOOLS
    even though they are pre-filtered out of the tools list seen by the model.
    Covers hallucinations where the model invents a tool name.

    Subagent names run a fresh chat_with_tools loop inside this call stack
    (scoped tools + prompt); the parent only sees the structured result the
    subagent returns per its own output schema.
    Every other name is a leaf tool dispatched from the registry."""
    if ctx.kind == "agent" and name not in AGENT_ALLOWED_TOOLS:
        logger.warning(
            "assistant-chat: rejected tool '%s' for agent (not in allowlist)", name,
        )
        return {"error": "tool_not_allowed_for_agent", "tool": name}

    spec = SUBAGENT_SPECS.get(name)
    if spec is not None:
        try:
            sub_args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            return {"error": "invalid_subagent_args"}
        if not isinstance(sub_args, dict):
            return {"error": "invalid_subagent_args"}
        query_value = str(sub_args.get(spec.input_field) or "").strip()
        if not query_value:
            return {"error": f"missing_{spec.input_field}"}
        # A subagent crash must not kill the parent turn: retry once, then
        # degrade to an error result the model can react to.
        for attempt in range(2):
            try:
                return _run_subagent(
                    spec, query_value, ctx.payload or {}, ctx.job_id,
                )
            except Exception:
                logger.exception(
                    "assistant-chat: subagent '%s' failed (attempt %d/2)",
                    name, attempt + 1,
                )
        return {"error": "subagent_failed", "tool": name}

    return execute_leaf(name, args_json, ctx)


# A subagent reply that should be a bare JSON object, possibly wrapped in a
# markdown fence — the only tolerated decoration. Anything else must fail so
# the schema-constrained retry kicks in instead of a lenient guess.
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


def _coerce_subagent_output(
    spec: SubagentSpec,
    llm,
    messages: List[Dict[str, Any]],
    content: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """Turn the subagent's final text into the JSON object its output schema
    promises.

    Fast path: the schema is in the system prompt, so the free reply is
    usually already that object. Otherwise regenerate once with the schema
    passed as `response_format` — llama-cpp constrains decoding to it, so the
    result is a conforming object by construction (of shape, not of values).
    The free reply is kept in context as a draft for that regeneration.
    Last resort (constrained call itself crashed): raw text as summary."""
    if spec.output_schema is None:
        return {"summary": content}
    required = spec.output_schema.get("required") or []
    parsed = _json_object_or_none(content)
    if parsed is not None and all(k in parsed for k in required):
        return parsed
    logger.info(
        "assistant-chat[sub:%s]: reply not schema-shaped, constraining decode",
        spec.name,
    )
    try:
        followup = list(messages)
        if content:
            followup.append({"role": "assistant", "content": content})
        followup.append({
            "role": "user",
            "content": (
                "Reply now with ONLY the JSON object matching the "
                "OUTPUT SCHEMA in your instructions. No other text."
            ),
        })
        forced = llm.chat(
            followup,
            max_tokens=max_tokens,
            response_format={"type": "json_object", "schema": spec.output_schema},
        ) or ""
        parsed = _json_object_or_none(_strip_thinking(forced))
        if parsed is not None:
            return parsed
    except Exception:
        logger.exception(
            "assistant-chat[sub:%s]: schema-constrained retry failed", spec.name,
        )
    return {"summary": content or ""}


def _run_subagent(
    spec: SubagentSpec,
    query: str,
    payload: Dict[str, Any],
    job_id: Optional[int],
) -> Dict[str, Any]:
    """Execute a subagent loop and return the object its output schema
    promises (both current specs: a `summary` plus structured fields like
    `sources` or `files`).

    Runs synchronously in the parent's call stack. Reuses the LLM singleton
    (no model reload thanks to (model_path, lora_path, lora_scale) caching)
    and emits its own logs prefixed by [sub:<name>] for correlation. Does NOT
    emit tool-events for its internal tool calls — only the parent's emit
    (running / done) wraps the subagent invocation. The subagent never sees
    the parent's conversation history; it starts from its own system prompt
    and the query it was handed. That is exactly the source of context
    compression for the parent.
    """
    cfg = get_task_config(spec.config_key)
    params = get_llm_params(spec.config_key)
    llm = get_llm_service(**params)
    max_tokens = int(cfg.get("max_tokens", spec.fallback_max_tokens))
    max_rounds = int(cfg.get("max_rounds", spec.max_rounds))

    # Filter ASSISTANT_TOOLS to the subagent's whitelist. Note that this
    # excludes any other subagent by construction — subagents cannot invoke
    # other subagents because no spec lists another subagent's name.
    tools_for_subagent = [
        t for t in ASSISTANT_TOOLS
        if t.get("function", {}).get("name") in spec.tool_names
    ]

    # Same date line the parent gets: subagents resolve temporal queries
    # ("next week") on their own, without the parent's context.
    now = datetime.now().astimezone().replace(microsecond=0)
    system_content = (
        f"{spec.system_prompt}\n\nCurrent date and time: {now.isoformat()} "
        f"({now:%A}). Resolve relative or time-only references like "
        "'at 10pm', 'tomorrow' or 'next Monday' against it."
    )

    # The subagent's own output schema: shown to the model so it knows the
    # exact object to return; the reply is parsed against it on exit.
    if spec.output_schema is not None:
        system_content = (
            f"{system_content}\n\nOUTPUT SCHEMA — once you stop calling "
            "tools, reply with a single JSON object matching this JSON "
            "Schema, and nothing else:\n"
            f"{json.dumps(spec.output_schema, ensure_ascii=False)}"
        )

    # Mirror the parent's /no_think convention so the subagent doesn't waste
    # tokens on visible reasoning.
    if not bool(cfg.get("enable_thinking", False)):
        system_content = f"{system_content}\n\n/no_think"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]

    logger.info(
        "assistant-chat[sub:%s]: enter — query=%r tools=%s rounds=%d",
        spec.name, query[:120], sorted(spec.tool_names), max_rounds,
    )

    kind = (payload.get("kind") or "assistant")
    owner_segment = _backend_owner_segment(payload)
    owner_id = _owner_id(payload)

    for round_idx in range(max_rounds):
        msg = llm.chat_with_tools(messages, tools_for_subagent, max_tokens=max_tokens)
        content = _strip_thinking(msg.get("content") or "")
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls and content:
            inline = _extract_inline_tool_calls(content)
            if inline:
                tool_calls = inline
        logger.info(
            "assistant-chat[sub:%s]: round %d → tool_calls=%d content_head=%r",
            spec.name, round_idx, len(tool_calls), content[:120],
        )
        if not tool_calls:
            logger.info("assistant-chat[sub:%s]: exit (final reply)", spec.name)
            return _coerce_subagent_output(spec, llm, messages, content, max_tokens)

        messages.append({
            "role": "assistant",
            "content": msg.get("content") or None,
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            fn = call.get("function") or {}
            tname = str(fn.get("name") or "")
            args_json = fn.get("arguments") or "{}"
            if tname not in spec.tool_names:
                result: Dict[str, Any] = {
                    "error": "tool_not_in_subagent_scope",
                    "tool": tname,
                }
            else:
                result = _execute_tool(
                    tname, args_json,
                    ToolContext(
                        kind=kind, owner_segment=owner_segment,
                        owner_id=owner_id, job_id=job_id, payload=payload,
                    ),
                )
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "name": tname,
                "content": json.dumps(result, ensure_ascii=False),
            })
            if isinstance(result, dict) and result.get("pendingConfirmation"):
                # Pending confirmation surfaced by an inner tool: the UI
                # already shows the card. The subagent must yield control
                # back to the parent so it can respond to the user.
                logger.info(
                    "assistant-chat[sub:%s]: pending confirmation from tool=%s, ending",
                    spec.name, tname,
                )
                return {
                    "summary": (
                        f"Awaiting user confirmation for {tname}. "
                        f"A confirmation card has been shown to the user."
                    ),
                }

    # Out of rounds: coerce a final reply with whatever the subagent
    # gathered. With an output schema, go straight to the grammar-forced
    # generation inside _coerce_subagent_output (empty content never parses).
    logger.info(
        "assistant-chat[sub:%s]: rounds exhausted, forcing final reply",
        spec.name,
    )
    if spec.output_schema is not None:
        return _coerce_subagent_output(spec, llm, messages, "", max_tokens)
    messages.append({
        "role": "user",
        "content": (
            "You have exhausted your tool budget. Reply now with what you "
            "found so far, in ≤200 words. Do not call any more tools."
        ),
    })
    final = llm.chat(messages, max_tokens=max_tokens, allow_thinking=True) or ""
    return {"summary": _strip_thinking(final)}


def _run_tool_rounds(
    llm,
    messages: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    payload: Dict[str, Any],
    job_id: Optional[int] = None,
) -> None:
    """Iterate tool-call rounds until the model produces a plain text reply or
    we hit MAX_TOOL_ROUNDS. Tool events are emitted LIVE via POST /tool-event
    (so the UI shows 'Searching...' the instant the tool starts) — this
    function no longer returns events to the caller."""
    tool_max_tokens = int(cfg.get("tool_max_tokens", 600))
    kind = (payload.get("kind") or "assistant")
    owner_id = _owner_id(payload)
    owner_segment = _backend_owner_segment(payload)
    can_emit_live = isinstance(owner_id, int) and isinstance(job_id, int)
    ctx = ToolContext(
        kind=kind, owner_segment=owner_segment,
        owner_id=owner_id, job_id=job_id, payload=payload,
    )
    tools_for_model = tools_for_payload(payload)
    tool_names = [t.get("function", {}).get("name") for t in tools_for_model]
    logger.info(
        "assistant-chat: tool round input kind=%s owner=%s/%s tools=%s",
        kind, owner_segment, owner_id, tool_names,
    )
    logger.info(
        "assistant-chat: tool round input — last_user=%r system_head=%r",
        next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")[:120],
        next((m.get("content") for m in messages if m.get("role") == "system"), "")[:160],
    )
    for round_idx in range(MAX_TOOL_ROUNDS):
        msg = llm.chat_with_tools(messages, tools_for_model, max_tokens=tool_max_tokens)
        content = _strip_thinking(msg.get("content") or "")
        tool_calls = msg.get("tool_calls") or []
        # Qwen3 often emits the call inline (<tool_call>…</tool_call>) instead
        # of populating tool_calls. Fall back to extracting from content.
        if not tool_calls and content:
            inline = _extract_inline_tool_calls(content)
            if inline:
                tool_calls = inline
        logger.info(
            "assistant-chat: tool round %d → tool_calls=%d content_head=%r",
            round_idx, len(tool_calls), content[:120],
        )
        if not tool_calls:
            # Model is ready to give the user a direct reply. We DON'T keep
            # this content — the final streaming call regenerates it.
            return
        # Persist the assistant message that triggered the calls — the
        # chat template needs it for the next turn to make sense.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or None,
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args_json = fn.get("arguments") or "{}"
            # Pull a short human-readable label from the args (e.g. the query).
            try:
                args_obj = json.loads(args_json or "{}")
                args_label = str(args_obj.get("query") or next(iter(args_obj.values()), ""))
            except (json.JSONDecodeError, StopIteration):
                args_label = ""
            # LIVE: emit the card the instant we know which tool the model
            # picked, before running it. The UI flips from "Thinking..." to
            # "Searching '<query>'..." immediately.
            if can_emit_live:
                post_tool_event(
                    owner_segment, owner_id, job_id, name, args_label,
                    status="running",
                )
            result = _execute_tool(name, args_json, ctx)
            # One-line summary + optional entity for the UI card; the per-tool
            # summarizers now live in the tools package.
            summary, entity = summarize_tool_result(name, result)
            logger.info("assistant-chat: tool=%s args=%s → %s",
                        name, args_json[:120], summary)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
            # LIVE: finalise the card with the result count + any created entity.
            # Tools that emit their own pending_confirmation card own the final
            # state — don't overwrite it with a generic `done` here.
            emitted_pending = (
                isinstance(result, dict) and result.get("pendingConfirmation")
            )
            if can_emit_live and not emitted_pending:
                post_tool_event(
                    owner_segment, owner_id, job_id, name, args_label,
                    status="done", summary=summary, entity=entity,
                )
    # Ran out of rounds. The streaming call will at least emit *something*
    # with the tool results in context.


@job_handler("assistant-chat")
def assistant_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cfg = get_task_config("assistant-chat")
        messages = _build_messages(payload, cfg)
        if not messages or messages[-1]["role"] != "user":
            return {"error": "History does not end with a user message"}

        max_tokens = int(cfg.get("max_tokens", 1000))
        params = get_llm_params("assistant-chat")
        llm = get_llm_service(**params)

        kind = (payload.get("kind") or "assistant")
        owner_id = _owner_id(payload)
        owner_segment = _backend_owner_segment(payload)

        logger.info(
            "assistant-chat: kind=%s owner=%s/%s name=%s turns=%d max_tokens=%d",
            kind, owner_segment, owner_id,
            payload.get("name") or payload.get("assistantName"),
            len(messages),
            max_tokens,
        )

        # IDs are needed for both tool live events and streaming chunks.
        job_id = payload.get("jobId")

        # Tool-call phase. The personal assistant gets the full ASSISTANT_TOOLS
        # list; agents get only the folder_* subset (filtered inside
        # _run_tool_rounds via tools_for_payload). The phase is non-streaming
        # because the model has to decide whether to call a tool BEFORE it
        # produces user-visible text. If a tool runs, its result gets appended
        # to `messages` and the streaming phase below sees the augmented
        # conversation as context. Tool cards are pushed LIVE to the UI via
        # POST /tool-event from inside _run_tool_rounds.
        #
        # Tool rounds are entered for: the personal assistant (assistantSystem)
        # and any agent (always — they only see folder_*, but that's exactly
        # what they need to do their job).
        if payload.get("assistantSystem") or kind == "agent":
            logger.info("assistant-chat: entering tool phase (kind=%s)", kind)
            _run_tool_rounds(llm, messages, cfg, payload, job_id)
        else:
            logger.info(
                "assistant-chat: skipping tool phase (assistantSystem=%r kind=%s)",
                payload.get("assistantSystem"), kind,
            )
        can_stream = (
            bool(cfg.get("stream", True))
            and isinstance(owner_id, int)
            and isinstance(job_id, int)
        )

        raw_parts: List[str] = []
        if can_stream:
            logger.info(
                "assistant-chat: streaming enabled (kind=%s owner=%s/%s job=%s)",
                kind, owner_segment, owner_id, job_id,
            )
            # Streaming state machine for <think>...</think> blocks: even with
            # `/no_think`, Qwen3 sometimes opens an empty pair at the start.
            # We accumulate everything in raw_parts (kept verbatim for the
            # final reply, where _strip_thinking handles it), but ONLY forward
            # chunks to the UI when we're outside any thinking block.
            visible_pending = ""        # characters seen but not yet decided
            in_think = False
            buffer: List[str] = []
            chunks_sent = 0
            last_flush_ms = time.monotonic() * 1000

            def _process_for_ui(piece: str) -> str:
                """Return the user-visible portion of this chunk, given current
                thinking state. Updates in_think / visible_pending across calls."""
                nonlocal in_think, visible_pending
                out_parts: List[str] = []
                buf = visible_pending + piece
                while buf:
                    if in_think:
                        end = buf.find("</think>")
                        if end == -1:
                            # Still inside thinking — swallow everything, keep
                            # nothing in pending (we never emit thinking text).
                            buf = ""
                            break
                        buf = buf[end + len("</think>"):]
                        in_think = False
                    else:
                        start = buf.find("<think>")
                        if start == -1:
                            # No new opening tag. But the tail might be a
                            # partial "<think" — hold it back to avoid leaking.
                            for i in range(1, min(len("<think>"), len(buf)) + 1):
                                if "<think>".startswith(buf[-i:]):
                                    out_parts.append(buf[:-i])
                                    visible_pending = buf[-i:]
                                    return "".join(out_parts)
                            out_parts.append(buf)
                            visible_pending = ""
                            return "".join(out_parts)
                        # Emit text before <think>, swallow the tag, enter thinking.
                        out_parts.append(buf[:start])
                        buf = buf[start + len("<think>"):]
                        in_think = True
                visible_pending = ""
                return "".join(out_parts)

            for piece in llm.chat_stream(messages, max_tokens=max_tokens):
                raw_parts.append(piece)
                visible = _process_for_ui(piece)
                if visible:
                    buffer.append(visible)
                now_ms = time.monotonic() * 1000
                if buffer and now_ms - last_flush_ms >= STREAM_FLUSH_INTERVAL_MS:
                    _post_stream_chunk(owner_segment, owner_id, job_id, "".join(buffer))
                    chunks_sent += 1
                    buffer.clear()
                    last_flush_ms = now_ms
            if buffer:
                _post_stream_chunk(owner_segment, owner_id, job_id, "".join(buffer))
                chunks_sent += 1
            # Final marker so the UI can stop the caret immediately, even
            # though memory extraction below may still take a beat.
            _post_stream_chunk(owner_segment, owner_id, job_id, "", done=True)
            raw = "".join(raw_parts)
            logger.info("assistant-chat: stream done, %d chunks sent", chunks_sent)
        else:
            logger.warning(
                "assistant-chat: streaming disabled (kind=%s owner=%s/%s job=%r stream_cfg=%r)",
                kind, owner_segment, owner_id, job_id, cfg.get("stream"),
            )
            raw = llm.chat(messages, max_tokens=max_tokens, allow_thinking=True) or ""

        reply = _strip_thinking(raw)
        if not reply:
            return {"error": "Model returned an empty response"}

        result: Dict[str, Any] = {"reply": reply}

        # Tool events are no longer returned here — they're emitted live by
        # _run_tool_rounds via POST /tool-event so the UI sees them in real
        # time, not at job completion.

        # Memory management is only relevant for the personal assistant.
        # Agents have no memory; skip the entire block.
        if kind == "assistant" and payload.get("assistantSystem"):
            user_message = _last_user_message(payload)
            if user_message:
                action = _extract_memory_action(
                    llm,
                    user_message,
                    payload.get("memorySnippets") or [],
                    cfg,
                )
                if action:
                    result["memoryAction"] = action
                    logger.info("assistant-chat: memoryAction=%r", action)

        return result
    except Exception as e:  # noqa: BLE001
        logger.exception("assistant-chat handler failed")
        return {"error": f"Assistant failure: {e}"}
