"""Chat-turn assembly shared by the chat task handlers.

Both the personal assistant (`core/tasks/assistant_chat`) and user-created agents
(`core/tasks/agent_chat`) build a turn the same way: one system message that
concatenates persona + multi-tool orientation + current date + (optionally)
persistent memory + the `/no_think` flag, an optional working-folder block, and
the tail of the conversation history. The two only differ in which persona and
orientation strings they inject and whether they carry a memory block — those are
parameters here.

This lives in `lib.llm` because it owns the LLM message shape; it knows nothing
about tasks, agents or the backend transport.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

# How many turns of history we keep as context. The backend persists the full
# thread, but passing it entirely to the LLM every call blows the context and
# spikes latency.
DEFAULT_HISTORY_TURNS = 16


def resolve_owner_id(
    payload: Dict[str, Any], keys: Sequence[str] = ("ownerId",),
) -> Optional[int]:
    """Resolve the owner id that addresses this turn's streaming / tool-event
    POSTs. Tries `keys` in order and returns the first int value; the caller
    passes its legacy alias as a fallback (e.g. `("ownerId", "assistantId")`)."""
    for key in keys:
        v = payload.get(key)
        if isinstance(v, int):
            return v
    return None


def build_chat_messages(
    payload: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    tool_orientation: str,
    base_prompt: str = "",
    default_system_prompt: str = "",
    memory_block: str = "",
    default_history_turns: int = DEFAULT_HISTORY_TURNS,
) -> List[Dict[str, str]]:
    """Assemble the LLM message list for one chat turn.

    The persona is the payload's custom `systemPrompt`, falling back to
    `default_system_prompt` when the payload carries none. `base_prompt` is a
    generic context prefix that is ALWAYS present and layered under the persona:
    a user-created agent, for instance, is defined by the user's `systemPrompt`
    but still needs its folder-tool context whatever that persona says. `memory_block`
    is the pre-formatted persistent-memory text (empty for handlers without
    memory, e.g. user agents); when present it is wrapped in the shared memory
    section.
    """
    raw_system_prompt = (payload.get("systemPrompt") or "").strip()
    # The persona comes from the payload (user-configured); the default only
    # applies when the payload carries none. The behaviour prompt lives in the
    # service, next to the tools — not in the backend.
    persona = raw_system_prompt or default_system_prompt
    system_prompt = "\n\n".join(p for p in (base_prompt, persona) if p)
    folder_scope = (payload.get("folderScope") or "").strip()
    conversation = payload.get("conversation") or []

    # Multi-tool composition orientation, concatenated into the same system
    # message (a second system message tends to be ignored).
    system_prompt = (
        f"{system_prompt}\n\n{tool_orientation}"
        if system_prompt
        else tool_orientation
    )

    now = datetime.now().astimezone().replace(microsecond=0)
    system_prompt = (
        f"{system_prompt}\n\nCurrent date and time: {now.isoformat()} "
        f"({now:%A}). Resolve relative or time-only references like "
        "'at 10pm', 'tomorrow' or 'next Monday' against it when scheduling."
    )

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

    # Modo de contexto del agente (opción configurable por agente creado por el
    # usuario). Un agente conversacional (p. ej. un redactor) ve el histórico
    # para arrastrar contexto entre turnos; uno no conversacional (p. ej. un
    # traductor) trata cada mensaje de forma independiente y solo debe recibir el
    # último. `conversational=False` colapsa la ventana a ese único turno.
    # Ausente => conversacional.
    if payload.get("conversational", True):
        history_turns = int(cfg.get("history_turns", default_history_turns))
    else:
        history_turns = 1
    filtered = [
        {"role": m["role"], "content": str(m.get("content") or "")}
        for m in conversation
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and m.get("content")
    ]
    messages.extend(filtered[-history_turns:])
    return messages
