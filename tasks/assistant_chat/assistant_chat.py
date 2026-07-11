"""Personal assistant handler.

Multi-turn chat with system prompt and history. This is NOT a Q&A over a file
(that's `ask`). The thread is persisted on the NestJS backend.

This module is a thin task handler: it builds the conversation (persona, memory,
date, working folder) and hands it to the personal assistant agent
(`core/agents`). The tool-calling loop, the tool repository and the subagents all
live outside this file; here we only assemble the turn and stream the reply.

Expected payload:
  {
    "ownerId": int,                       # id in owner's table
    "name": str,                          # owner's display name
    "systemPrompt": str | null,           # owner's custom prompt; null => default
    "folderScope": str | null,            # working folder, passed through to tools
    "assistantId": int,                   # legacy alias of ownerId
    "assistantName": str,                 # legacy alias of name
    "assistantSystem": bool,              # true => run the tool phase
    "memorySnippets": [...],              # injected memory
    "extractMemory": bool,                # run memory extraction
    "conversation": [{"role": ..., "content": ...}, ...]
  }

Returns:
  {"reply": str, "memoryAction"?: {...}}  or  {"error": str}
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import urllib.error
import urllib.request

from services.llm_service import get_llm_service
from services.model_config import get_llm_params, get_task_config
from services.prompts import load_prompt
from utils.job_registry import job_handler

from tools import ToolContext
from agents import MAIN_AGENT, run_agent
from common.chat.http import BACKEND_URL
from common.chat.text_utils import strip_thinking as _strip_thinking
from tasks.assistant_chat.memory_agent import (
    extract_memory_action as _extract_memory_action,
    format_memory_block as _format_memory_block,
    last_user_message as _last_user_message,
    memory_for_payload as _memory_for_payload,
)

logger = logging.getLogger(__name__)

# indexed-files / stream-chunk endpoints are always under /assistants/:id.
OWNER_SEGMENT = "assistants"


def _owner_id(payload: Dict[str, Any]) -> Optional[int]:
    """Resolve the owner id for streaming / tool-event POSTs.
    Prefers `ownerId`; falls back to the legacy `assistantId`."""
    for key in ("ownerId", "assistantId"):
        v = payload.get(key)
        if isinstance(v, int):
            return v
    return None


_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

DEFAULT_ASSISTANT_SYSTEM_PROMPT = load_prompt(_PROMPTS_DIR, "assistant_system.md").strip()

# Multi-tool composition orientation, injected into the system prompt: compose,
# chain, don't repeat, stop when done.
MULTI_TOOL_ORIENTATION = load_prompt(_PROMPTS_DIR, "multi_tool_orientation.md").strip()


# Flush a partial chunk to the backend roughly every N ms, regardless of how
# many tokens have accumulated. Tuned so the user sees forward motion without
# drowning the HTTP loop in tiny requests.
STREAM_FLUSH_INTERVAL_MS = 120

# How many turns of history we keep as context. The backend persists the full
# thread, but passing it entirely to the LLM every call blows the context and
# spikes latency.
DEFAULT_HISTORY_TURNS = 16


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

    When `done=True`, signals that the model has finished generating even if the
    job hasn't fully completed (memory extraction may still run). The UI uses it
    to stop the live caret immediately."""
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


def _build_messages(payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_system_prompt = (payload.get("systemPrompt") or "").strip()
    # No custom prompt → use this service's default. The behaviour prompt lives
    # here, next to the tools — not in the backend.
    system_prompt = raw_system_prompt or DEFAULT_ASSISTANT_SYSTEM_PROMPT
    folder_scope = (payload.get("folderScope") or "").strip()
    memory_snippets = _memory_for_payload(payload)
    conversation = payload.get("conversation") or []

    # Multi-tool composition orientation, concatenated into the same system
    # message (a second system message tends to be ignored), between the persona
    # and the persistent memory block.
    system_prompt = (
        f"{system_prompt}\n\n{MULTI_TOOL_ORIENTATION}"
        if system_prompt
        else MULTI_TOOL_ORIENTATION
    )

    now = datetime.now().astimezone().replace(microsecond=0)
    system_prompt = (
        f"{system_prompt}\n\nCurrent date and time: {now.isoformat()} "
        f"({now:%A}). Resolve relative or time-only references like "
        "'at 10pm', 'tomorrow' or 'next Monday' against it when scheduling."
    )

    memory_block = _format_memory_block(memory_snippets)
    if memory_block:
        memory_section = (
            "What you already know about the user (persistent memory from "
            "previous conversations). Lean on these facts when they are "
            "relevant to your answer; if the user asks something whose answer "
            "is here, use it directly:\n"
            f"{memory_block}"
        )
        system_prompt = f"{system_prompt}\n\n{memory_section}"

    # Qwen3 emits <think>...</think> before every response by default. That
    # consumes tokens (high latency) and adds no value for conversational chat;
    # disable it via the official `/no_think` flag. Configurable in case we want
    # explicit thinking in another phase.
    if not bool(cfg.get("enable_thinking", False)):
        system_prompt = f"{system_prompt}\n\n/no_think"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Working folder as a separate context block, so the model can mention it and
    # folder-aware tools receive its path via the ToolContext.
    if folder_scope:
        messages.append({
            "role": "system",
            "content": f"[WORKING FOLDER]\nPath: {folder_scope}",
        })

    history_turns = int(cfg.get("history_turns", DEFAULT_HISTORY_TURNS))
    filtered = [
        {"role": m["role"], "content": str(m.get("content") or "")}
        for m in conversation
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and m.get("content")
    ]
    messages.extend(filtered[-history_turns:])
    return messages


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

        owner_id = _owner_id(payload)
        job_id = payload.get("jobId")
        folder_scope = (payload.get("folderScope") or "").strip()

        logger.info(
            "assistant-chat: owner=%s/%s name=%s turns=%d max_tokens=%d",
            OWNER_SEGMENT, owner_id,
            payload.get("name") or payload.get("assistantName"),
            len(messages), max_tokens,
        )

        ctx = ToolContext(
            owner_segment=OWNER_SEGMENT, owner_id=owner_id, job_id=job_id,
            folder_scope=folder_scope, payload=payload,
        )

        # Tool-call phase (non-streaming: the model has to decide whether to call
        # a tool BEFORE it produces user-visible text). If a tool runs, its
        # result is appended to `messages` and the streaming phase below sees the
        # augmented conversation. Tool cards are pushed LIVE via POST /tool-event
        # from inside the agent loop.
        if payload.get("assistantSystem"):
            logger.info("assistant-chat: entering tool phase")
            run_agent(MAIN_AGENT, messages, ctx)
        else:
            logger.info("assistant-chat: skipping tool phase (assistantSystem falsy)")

        can_stream = (
            bool(cfg.get("stream", True))
            and isinstance(owner_id, int)
            and isinstance(job_id, int)
        )

        raw_parts: List[str] = []
        if can_stream:
            logger.info("assistant-chat: streaming (owner=%s/%s job=%s)",
                        OWNER_SEGMENT, owner_id, job_id)
            # Streaming state machine for <think>...</think> blocks: even with
            # `/no_think`, Qwen3 sometimes opens an empty pair at the start. We
            # accumulate everything in raw_parts (kept verbatim for the final
            # reply, where _strip_thinking handles it), but ONLY forward chunks
            # to the UI when we're outside any thinking block.
            visible_pending = ""
            in_think = False
            buffer: List[str] = []
            chunks_sent = 0
            last_flush_ms = time.monotonic() * 1000

            def _process_for_ui(piece: str) -> str:
                """Return the user-visible portion of this chunk given current
                thinking state. Updates in_think / visible_pending across calls."""
                nonlocal in_think, visible_pending
                out_parts: List[str] = []
                buf = visible_pending + piece
                while buf:
                    if in_think:
                        end = buf.find("</think>")
                        if end == -1:
                            buf = ""
                            break
                        buf = buf[end + len("</think>"):]
                        in_think = False
                    else:
                        start = buf.find("<think>")
                        if start == -1:
                            for i in range(1, min(len("<think>"), len(buf)) + 1):
                                if "<think>".startswith(buf[-i:]):
                                    out_parts.append(buf[:-i])
                                    visible_pending = buf[-i:]
                                    return "".join(out_parts)
                            out_parts.append(buf)
                            visible_pending = ""
                            return "".join(out_parts)
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
                    _post_stream_chunk(OWNER_SEGMENT, owner_id, job_id, "".join(buffer))
                    chunks_sent += 1
                    buffer.clear()
                    last_flush_ms = now_ms
            if buffer:
                _post_stream_chunk(OWNER_SEGMENT, owner_id, job_id, "".join(buffer))
                chunks_sent += 1
            _post_stream_chunk(OWNER_SEGMENT, owner_id, job_id, "", done=True)
            raw = "".join(raw_parts)
            logger.info("assistant-chat: stream done, %d chunks sent", chunks_sent)
        else:
            logger.warning(
                "assistant-chat: streaming disabled (owner=%s/%s job=%r stream_cfg=%r)",
                OWNER_SEGMENT, owner_id, job_id, cfg.get("stream"),
            )
            raw = llm.chat(messages, max_tokens=max_tokens, allow_thinking=True) or ""

        reply = _strip_thinking(raw)
        if not reply:
            return {"error": "Model returned an empty response"}

        result: Dict[str, Any] = {"reply": reply}

        # Persistent user memory: extract after replying.
        if payload.get("assistantSystem"):
            user_message = _last_user_message(payload)
            if user_message:
                action = _extract_memory_action(
                    llm, user_message, payload.get("memorySnippets") or [], cfg,
                )
                if action:
                    result["memoryAction"] = action
                    logger.info("assistant-chat: memoryAction=%r", action)

        return result
    except Exception as e:  # noqa: BLE001
        logger.exception("assistant-chat handler failed")
        return {"error": f"Assistant failure: {e}"}
