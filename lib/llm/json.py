"""
JSON-from-LLM helpers.

Local llama-cpp models do not have a reliable native function-calling API,
so we extract JSON from free-form chat responses with a tolerant parser
and a one-retry-with-feedback wrapper.
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    return text


def _extract_outermost(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the outermost balanced span starting with open_ch, or None."""
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json(text: str, default: Any = None) -> Any:
    """
    Tolerant JSON extractor. Strips ```json fences, then tries:
      1) json.loads on the stripped text.
      2) outermost {...} balanced span.
      3) outermost [...] balanced span.
    Returns the default if everything fails.
    """
    if not text:
        return default
    stripped = _strip_fences(text).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        span = _extract_outermost(stripped, opener, closer)
        if span:
            try:
                return json.loads(span)
            except json.JSONDecodeError:
                continue

    return default


def chat_json(
    llm,
    messages: list,
    schema_hint: str,
    max_retries: int = 2,
    max_tokens: int = 500,
) -> Any:
    """
    Call llm.chat and parse JSON from the response. On parse failure,
    re-ask up to max_retries times with the previous (invalid) response
    plus an explicit reminder of the expected schema.

    Returns the parsed JSON value, or None if all retries fail.
    """
    convo = list(messages)
    last_raw = None

    for attempt in range(max_retries + 1):
        raw = llm.chat(convo, max_tokens=max_tokens)
        last_raw = raw
        parsed = parse_json(raw, default=None)
        if parsed is not None:
            return parsed

        logger.warning(
            "chat_json: parse failed on attempt %d/%d. raw=%r",
            attempt + 1, max_retries + 1, raw[:200] if raw else "",
        )
        convo = list(messages) + [
            {"role": "assistant", "content": raw or ""},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. Respond ONLY "
                    "with a JSON value matching this schema, no prose, no fences:\n"
                    + schema_hint
                ),
            },
        ]

    logger.error("chat_json: exhausted retries. last_raw=%r", (last_raw or "")[:200])
    return None
