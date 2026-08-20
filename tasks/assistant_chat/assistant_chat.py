"""Personal assistant handler.

Multi-turn chat with system prompt and history. This is NOT a Q&A over a file
(that's `ask`). The thread is persisted on the NestJS backend.

This handler is the PERSONAL ASSISTANT only: the built-in, memory-backed agent
with the workspace toolset (tasks, calendar, notes, workspace research). Chat
with a user-created agent is a different execution type — see the `agent-chat` task
(`core/tasks/agent_chat`) — even though the backend distinguishes them by
`payload.kind`.

This module is a thin task handler: it builds the conversation (persona, memory,
date, working folder) and hands it to the personal assistant agent
(`core/agents`). The tool-calling loop, the tool repository and the subagents all
live outside this file; here we assemble the turn and return its final reply.

Expected payload:
  {
    "ownerId": int,                       # id in owner's table
    "name": str,                          # owner's display name
    "systemPrompt": str | null,           # owner's custom prompt; null => default
    "folderScope": str | null,            # working folder, passed through to tools
    "assistantSystem": bool,              # true => tools and persistent memory
    "memorySnippets": [...] | null,       # optional extraction context
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
from common.execution_registry import execution_handler

from tools import ToolContext
from agents import assistant
from agents.assistant import DEFAULT_SYSTEM_PROMPT, MULTI_TOOL_ORIENTATION
from agents.memory_agent import (
    extract_memory_action as _extract_memory_action,
    format_memory_block as _format_memory_block,
    last_user_message as _last_user_message,
    memory_for_payload as _memory_for_payload,
)
from lib.execution import (
    ExecutionEmitter,
    InferenceBudgetDenied,
    ProgressLoopContext,
    activate_emitter,
    reset_emitter,
)

logger = logging.getLogger(__name__)

# indexed-files / stream-chunk endpoints are always under /assistants/:id.
OWNER_SEGMENT = "assistants"


@execution_handler("assistant-chat")
def assistant_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    emitter = ExecutionEmitter.from_payload(payload)
    emitter_token = activate_emitter(emitter)
    try:
        cfg = get_task_config("assistant-chat")
        messages = build_chat_messages(
            payload, cfg,
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            tool_orientation=MULTI_TOOL_ORIENTATION,
            memory_block=_format_memory_block(_memory_for_payload(payload)),
        )
        if not messages or messages[-1]["role"] != "user":
            error = "History does not end with a user message"
            emitter.flush_evidence()
            return emitter.attach_summary({"error": error})

        max_tokens = int(cfg.get("max_tokens", 1000))
        params = get_llm_params("assistant-chat")
        llm = get_llm_service(**params)

        owner_id = resolve_owner_id(payload)
        execution_id = payload.get("executionId")
        folder_scope = (payload.get("folderScope") or "").strip()

        logger.info(
            "assistant-chat: owner=%s/%s name=%s turns=%d max_tokens=%d",
            OWNER_SEGMENT, owner_id,
            payload.get("name"),
            len(messages), max_tokens,
        )

        ctx = ToolContext(
            owner_segment=OWNER_SEGMENT, owner_id=owner_id, execution_id=execution_id,
            folder_scope=folder_scope, payload=payload,
            execution=emitter,
        )

        effective_tools = assistant.tools(ctx) if payload.get("assistantSystem") else []
        if effective_tools:
            logger.info("assistant-chat: entering tool phase")
            outcome = assistant.run(messages, ctx)
            if outcome.kind == "invalid":
                reason = outcome.reason or "invalid_agent_result"
                if reason.startswith("budget_"):
                    emitter.flush_evidence()
                    return emitter.attach_summary({
                        "error": reason,
                        "completionReason": "budget_exhausted",
                    })
            raw = outcome.content if outcome.kind == "final_text" else ""
        else:
            logger.info("assistant-chat: direct response without effective tools")
            direct_progress = ProgressLoopContext.start(
                emitter,
                agent_name="assistant",
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

        result: Dict[str, Any] = {"reply": reply}
        if effective_tools and outcome.completion_kind:
            result["completionKind"] = outcome.completion_kind
            result["completionReason"] = outcome.completion_reason
        user_message = ""
        memory_trace = None

        if payload.get("assistantSystem"):
            user_message = _last_user_message(payload)
            if user_message:
                memory_max_tokens = int(cfg.get("memory_extract_max_tokens", 256))
                memory_progress = ProgressLoopContext.start(
                    emitter,
                    agent_name="memory_agent",
                    loop_kind="synchronous_subagent",
                    max_rounds=1,
                    max_output_repairs=0,
                    forced_finalization_available=False,
                    max_tokens_per_inference=memory_max_tokens,
                )
                memory_trace = memory_progress.trace(
                    round=1,
                    phase="memory_extraction",
                )

        emitter.record_final_message(reply)

        # Persistent user memory: extract after the final response is durable.
        if user_message and memory_trace:
            action = _extract_memory_action(
                llm,
                user_message,
                payload.get("memorySnippets") or [],
                cfg,
                trace_metadata=memory_trace,
            )
            if action:
                result["memoryAction"] = action
                logger.info("assistant-chat: memoryAction=%r", action)

        emitter.flush_evidence()
        return emitter.attach_summary(result)
    except InferenceBudgetDenied as e:
        emitter.flush_evidence()
        return emitter.attach_summary({
            "error": e.reason,
            "completionReason": "budget_exhausted",
        })
    except Exception as e:  # noqa: BLE001
        logger.exception("assistant-chat handler failed")
        error = f"Assistant failure: {e}"
        emitter.flush_evidence()
        return emitter.attach_summary({"error": error})
    finally:
        reset_emitter(emitter_token)
