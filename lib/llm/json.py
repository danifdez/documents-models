"""
JSON-from-LLM helpers.

Small local models do not reliably answer with the JSON they were asked for, so
we extract JSON from free-form chat responses with a tolerant parser and a
one-retry-with-feedback wrapper. (When the output shape matters, prefer
`chat(response_format=...)`: the engine compiles the schema to a grammar and
constrains decoding, which cannot produce invalid JSON in the first place.)
"""

import json
import re
from typing import Any, Optional

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
