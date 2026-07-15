"""User-created agent handler.

Multi-turn chat with a user-created agent: a custom persona (`systemPrompt`)
scoped to a working folder (`folderScope`), with the folder toolset. It is the
sibling of the personal-assistant handler (`core/tasks/assistant_chat`); the
backend distinguishes the two by `payload.kind` ('agent' vs 'assistant') but they
are separate jobs and separate responsibilities:

- the assistant is the built-in, memory-backed agent with the workspace toolset;
- a user agent has NO personal-assistant memory and only its folder tools.

This module is a thin task handler: it builds the conversation (persona,
orientation, date, working folder) and drives the user agent (`core/agents`). The
tool-calling loop and the tool repository live outside this file; here we only
assemble the turn and stream the reply. Tool cards are pushed LIVE via POST
/agents/:id/tool-event from inside the agent loop.

Expected payload (built by the backend's AgentService):
  {
    "kind": "agent",
    "ownerId": int,                       # agent id
    "agentId": int,                       # explicit agent id (same value)
    "assistantName": str,                 # agent display name
    "systemPrompt": str | null,           # agent's persona; null => default
    "folderScope": str | null,            # working folder, passed through to tools
    "conversational": bool,               # agent option: true (default) => chat
                                          #   with prior turns as context; false =>
                                          #   stateless, only the last message is
                                          #   sent (e.g. a translator)
    "conversation": [{"role": ..., "content": ...}, ...]
  }

Returns:
  {"reply": str}  or  {"error": str}
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
from agents.user_agent import (
    BASE_SYSTEM_PROMPT,
    MULTI_TOOL_ORIENTATION,
    user_agent_for,
)

logger = logging.getLogger(__name__)

# stream-chunk / tool-event / indexed-files endpoints are under /agents/:id.
OWNER_SEGMENT = "agents"


@job_handler("agent-chat")
def agent_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cfg = get_task_config("agent-chat")
        messages = build_chat_messages(
            payload, cfg,
            base_prompt=BASE_SYSTEM_PROMPT,
            tool_orientation=MULTI_TOOL_ORIENTATION,
        )
        if not messages or messages[-1]["role"] != "user":
            return {"error": "History does not end with a user message"}

        max_tokens = int(cfg.get("max_tokens", 1000))
        params = get_llm_params("agent-chat")
        llm = get_llm_service(**params)

        owner_id = resolve_owner_id(payload, ("ownerId", "agentId"))
        job_id = payload.get("jobId")
        folder_scope = (payload.get("folderScope") or "").strip()

        logger.info(
            "agent-chat: owner=%s/%s name=%s turns=%d max_tokens=%d",
            OWNER_SEGMENT, owner_id, payload.get("assistantName"),
            len(messages), max_tokens,
        )

        ctx = ToolContext(
            owner_segment=OWNER_SEGMENT, owner_id=owner_id, job_id=job_id,
            folder_scope=folder_scope, payload=payload,
        )

        # Tool-call phase (non-streaming: the model has to decide whether to call
        # a tool BEFORE it produces user-visible text). If a tool runs, its
        # result is appended to `messages` and the streaming phase below sees the
        # augmented conversation.
        logger.info("agent-chat: entering tool phase")
        user_agent_for(payload).run(messages, ctx)

        raw = generate_reply(
            llm, messages, max_tokens,
            owner_segment=OWNER_SEGMENT, owner_id=owner_id, job_id=job_id,
            stream_enabled=bool(cfg.get("stream", True)),
        )

        reply = _strip_thinking(raw)
        if not reply:
            return {"error": "Model returned an empty response"}

        return {"reply": reply}
    except Exception as e:  # noqa: BLE001
        logger.exception("agent-chat handler failed")
        return {"error": f"Agent failure: {e}"}
