"""User-created agent handler.

Multi-turn chat with a user-created agent: a custom persona (`systemPrompt`)
scoped to a working folder (`folderScope`), with the folder toolset. It is the
sibling of the personal-assistant handler (`core/tasks/assistant_chat`); the
backend distinguishes the two by `payload.kind` ('agent' vs 'assistant') but they
are separate execution types and responsibilities:

- the assistant is the built-in, memory-backed agent with the workspace toolset;
- a user agent has NO personal-assistant memory and only its folder tools.

This module is a thin task handler: it builds the conversation (persona,
orientation, date, working folder) and drives the user agent (`core/agents`). The
tool-calling loop and the tool repository live outside this file; here we
assemble the turn and return its final reply. Tool cards are pushed LIVE via POST
/agents/:id/tool-event from inside the agent loop.

Expected payload (built by the backend's AgentService):
  {
    "kind": "agent",
    "ownerId": int,                       # agent id
    "name": str,                          # agent display name
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
from common.execution_registry import execution_handler

from tools import ToolContext
from agents.user_agent import (
    BASE_SYSTEM_PROMPT,
    MULTI_TOOL_ORIENTATION,
    user_agent_for,
)
from lib.execution import (
    ExecutionEmitter,
    ProgressLoopContext,
    activate_emitter,
    reset_emitter,
)

logger = logging.getLogger(__name__)

# stream-chunk / tool-event / indexed-files endpoints are under /agents/:id.
OWNER_SEGMENT = "agents"


@execution_handler("agent-chat")
def agent_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    emitter = ExecutionEmitter.from_payload(payload)
    emitter_token = activate_emitter(emitter)
    try:
        cfg = get_task_config("agent-chat")
        messages = build_chat_messages(
            payload, cfg,
            base_prompt=BASE_SYSTEM_PROMPT,
            tool_orientation=MULTI_TOOL_ORIENTATION,
        )
        if not messages or messages[-1]["role"] != "user":
            error = "History does not end with a user message"
            emitter.flush_evidence()
            return emitter.attach_summary({"error": error})

        max_tokens = int(cfg.get("max_tokens", 1000))
        params = get_llm_params("agent-chat")
        llm = get_llm_service(**params)

        owner_id = resolve_owner_id(payload)
        execution_id = payload.get("executionId")
        folder_scope = (payload.get("folderScope") or "").strip()

        logger.info(
            "agent-chat: owner=%s/%s name=%s turns=%d max_tokens=%d",
            OWNER_SEGMENT, owner_id, payload.get("name"),
            len(messages), max_tokens,
        )

        ctx = ToolContext(
            owner_segment=OWNER_SEGMENT, owner_id=owner_id, execution_id=execution_id,
            folder_scope=folder_scope, payload=payload,
            execution=emitter,
        )

        agent = user_agent_for(payload)
        if agent.tools(ctx):
            logger.info("agent-chat: entering tool phase")
            outcome = agent.run(messages, ctx)
            raw = outcome.content if outcome.kind == "final_text" else ""
        else:
            logger.info("agent-chat: direct response without effective tools")
            direct_progress = ProgressLoopContext.start(
                emitter,
                agent_name="user_agent",
                loop_kind="top_level",
                max_rounds=1,
                max_output_repairs=0,
                forced_finalization_available=False,
                max_tokens_per_inference=max_tokens,
            )
            raw = generate_reply(
                llm, messages, max_tokens,
                owner_segment=OWNER_SEGMENT, owner_id=owner_id, execution_id=execution_id,
                stream_enabled=bool(cfg.get("stream", True)),
                inference_name="direct_response",
                trace_metadata=direct_progress.trace(
                    round=1,
                    phase="direct_response",
                ),
            )

        reply = _strip_thinking(raw)
        if not reply:
            error = "Model returned an empty response"
            emitter.flush_evidence()
            return emitter.attach_summary({"error": error})

        emitter.record_final_message(reply)
        emitter.flush_evidence()
        return emitter.attach_summary({"reply": reply})
    except Exception as e:  # noqa: BLE001
        logger.exception("agent-chat handler failed")
        error = f"Agent failure: {e}"
        emitter.flush_evidence()
        return emitter.attach_summary({"error": error})
    finally:
        reset_emitter(emitter_token)
