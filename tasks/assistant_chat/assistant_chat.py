"""Personal assistant handler.

Multi-turn chat with system prompt and history. This is NOT a Q&A over a file
(that's `ask`). The thread is persisted on the NestJS backend.

This handler is the PERSONAL ASSISTANT only: the built-in, memory-backed agent
with the workspace toolset (tasks, calendar, notes, workspace research). Chat
with a user-created agent is a different job — see the `agent-chat` task
(`core/tasks/agent_chat`) — even though the backend distinguishes them by
`payload.kind`.

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

import logging
from typing import Any, Dict

from services.llm_service import get_llm_service
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.chat import build_chat_messages, resolve_owner_id
from lib.llm.text import strip_thinking as _strip_thinking
from lib.backend.stream import generate_reply
from common.job_registry import job_handler

from tools import ToolContext
from agents import assistant
from agents.assistant import DEFAULT_SYSTEM_PROMPT, MULTI_TOOL_ORIENTATION
from agents.memory_agent import (
    extract_memory_action as _extract_memory_action,
    format_memory_block as _format_memory_block,
    last_user_message as _last_user_message,
    memory_for_payload as _memory_for_payload,
)

logger = logging.getLogger(__name__)

# indexed-files / stream-chunk endpoints are always under /assistants/:id.
OWNER_SEGMENT = "assistants"


@job_handler("assistant-chat")
def assistant_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cfg = get_task_config("assistant-chat")
        messages = build_chat_messages(
            payload, cfg,
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            tool_orientation=MULTI_TOOL_ORIENTATION,
            memory_block=_format_memory_block(_memory_for_payload(payload)),
        )
        if not messages or messages[-1]["role"] != "user":
            return {"error": "History does not end with a user message"}

        max_tokens = int(cfg.get("max_tokens", 1000))
        params = get_llm_params("assistant-chat")
        llm = get_llm_service(**params)

        owner_id = resolve_owner_id(payload, ("ownerId", "assistantId"))
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
            assistant.run(messages, ctx)
        else:
            logger.info("assistant-chat: skipping tool phase (assistantSystem falsy)")

        raw = generate_reply(
            llm, messages, max_tokens,
            owner_segment=OWNER_SEGMENT, owner_id=owner_id, job_id=job_id,
            stream_enabled=bool(cfg.get("stream", True)),
        )

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
