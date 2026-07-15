"""Persistent-memory agent for the personal assistant.

Two responsibilities, both driven by the job payload:
  - injection: pull the memories relevant to the user's message into the
    system prompt (`memory_for_payload` / `format_memory_block`);
  - extraction: a second LLM call that decides whether to save/forget/replace
    a memory from the user's message (`extract_memory_action`).

This is a mini-agent, not a tool: the model never calls it — the handler runs
it after every personal-assistant turn.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from lib.llm.text import strip_thinking
from lib.llm.prompts import load_prompt

logger = logging.getLogger(__name__)


# Valid types for a memory entry. Mirrors the backend enum.
# Three canonical categories (semantic / episodic / procedural in cognitive
# science): `fact` covers stable knowledge about the user or the world
# (includes preferences, data, identity, tools, links); `episode` is a
# narrative memory of something that happened or that the assistant should
# know contextually (does NOT replace the calendar — things that need an
# alarm/slot go there); `instruction` is how the assistant should behave.
MEMORY_TYPES = {"fact", "episode", "instruction"}

# Structured extraction/management prompt. Called on every turn of the
# personal assistant. The LLM decides among four actions — `save` (save a
# new memory), `forget` (forget an existing one by id), `replace` (correct
# the value of an existing memory by id) or `none` (nothing to do) — and
# returns ONE JSON with that decision. Body in prompts/memory_extract.md.
_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
EXTRACT_MEMORY_PROMPT = load_prompt(_PROMPTS_DIR, "memory_extract.md")


def last_user_message(payload: Dict[str, Any]) -> Optional[str]:
    conversation = payload.get("conversation") or []
    for m in reversed(conversation):
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            return str(m["content"]).strip()
    return None


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Accepts either clean JSON or a JSON surrounded by prose. Returns None
    if no valid object can be extracted."""
    if not text:
        return None
    cleaned = strip_thinking(text)
    # Direct attempt
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Look for the first brace-block within the text
    m = _JSON_OBJECT_RE.search(cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _format_memory_for_prompt(snippets: List[Dict[str, Any]]) -> str:
    """Render existing memory as a numbered list with ids — the LLM uses these
    ids to point at entries to forget or replace. The `relevance` hint
    (high/medium/recent) helps the model decide whether a candidate is
    semantically related to the user's message."""
    lines: List[str] = []
    for s in snippets:
        if not isinstance(s, dict):
            continue
        mid = s.get("id")
        name = (s.get("name") or "").strip()
        type_ = (s.get("type") or "other").strip()
        body = (s.get("body") or "").strip()
        relevance = (s.get("relevance") or "recent").strip().lower()
        if mid is None or not name or not body:
            continue
        lines.append(f"- id={mid} ({type_}, {relevance}) {name}: {body}")
    return "\n".join(lines) if lines else "(no memory yet)"


def extract_memory_action(
    llm,
    user_message: str,
    memory_snippets: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Second LLM call: decide save/forget/none based on the user message and
    existing memory. Returns a dict shaped like::

        {"action": "save", "save": {name, type, body}}
        {"action": "forget", "forget_id": int}
        None  # for action=none, parse errors, or invalid output

    The caller (backend processor) reacts to the action accordingly. Memory
    ids in `memory_snippets` are the authoritative ones from the DB — the
    forget_id returned by the LLM must match one of them.
    """
    prompt = EXTRACT_MEMORY_PROMPT.format(
        message=user_message,
        memory_list=_format_memory_for_prompt(memory_snippets),
    )
    # Thinking OFF by default: this runs after every reply on the single local
    # worker and the rule-guided choice rarely needs it. Re-enable (and raise
    # max_tokens) via config if memory quality drops.
    extract_thinking = bool(cfg.get("memory_extract_thinking", False))
    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
    try:
        raw = llm.chat(
            messages,
            max_tokens=int(cfg.get("memory_extract_max_tokens", 256)),
            allow_thinking=extract_thinking,
        ) or ""
    except Exception:
        logger.exception("assistant-chat: memory extraction failed")
        return None

    obj = _parse_json_object(raw)
    if not obj:
        logger.info("assistant-chat: could not parse memory JSON: %r", raw[:200])
        return None

    action = str(obj.get("action") or "none").strip().lower()

    if action == "save":
        name = str(obj.get("name") or "").strip()
        body = str(obj.get("body") or "").strip()
        type_ = str(obj.get("type") or "fact").strip().lower()
        if type_ not in MEMORY_TYPES:
            type_ = "fact"
        if not name or not body:
            return None
        return {"action": "save", "save": {"name": name[:120], "type": type_, "body": body}}

    if action == "replace":
        name = str(obj.get("name") or "").strip()
        body = str(obj.get("body") or "").strip()
        type_ = str(obj.get("type") or "fact").strip().lower()
        if type_ not in MEMORY_TYPES:
            type_ = "fact"
        try:
            replace_id = int(obj.get("replace_id"))
        except (TypeError, ValueError):
            logger.info("assistant-chat: replace without valid replace_id")
            return None
        known_ids = {s.get("id") for s in memory_snippets if isinstance(s, dict)}
        if replace_id not in known_ids:
            logger.info("assistant-chat: replace_id %r is not in known memory", replace_id)
            return None
        if not name or not body:
            return None
        return {
            "action": "replace",
            "replace_id": replace_id,
            "save": {"name": name[:120], "type": type_, "body": body},
        }

    if action == "forget":
        try:
            forget_id = int(obj.get("forget_id"))
        except (TypeError, ValueError):
            return None
        # Validate against the snippets we sent — never trust the LLM to
        # invent ids it didn't see.
        known_ids = {s.get("id") for s in memory_snippets if isinstance(s, dict)}
        if forget_id not in known_ids:
            logger.info("assistant-chat: forget_id %r is not in known memory", forget_id)
            return None
        return {"action": "forget", "forget_id": forget_id}

    return None


def format_memory_block(snippets: List[Dict[str, Any]]) -> str:
    """Format injected memory entries as a single context block.

    Returns an empty string if there is nothing to inject. Caller is expected
    to embed the result into the main system prompt with explicit framing so
    the model actually uses it (a separate system message often gets ignored).
    """
    lines: List[str] = []
    for s in snippets:
        if not isinstance(s, dict):
            continue
        name = (s.get("name") or "").strip()
        type_ = (s.get("type") or "other").strip()
        body = (s.get("body") or "").strip()
        if not name or not body:
            continue
        lines.append(f"- ({type_}) {name}: {body}")
    return "\n".join(lines)


def memory_for_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not payload.get("assistantSystem"):
        return []
    assistant_id = payload.get("assistantId")
    if assistant_id is None:
        return []
    from tasks.memory.memory import relevant_for_injection
    try:
        return relevant_for_injection(int(assistant_id), last_user_message(payload) or "")
    except Exception:
        logger.exception("assistant-chat: memory injection failed")
        return []
