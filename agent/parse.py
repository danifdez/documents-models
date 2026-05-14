"""Parses an agent's LLM decision into a structured action."""

import logging
from typing import Any, Dict, Optional

from services.llm_json import parse_json

logger = logging.getLogger(__name__)


def parse_decision(raw: str) -> Optional[Dict[str, Any]]:
    """
    Expects the model to emit JSON of one of these shapes:
      {"thought": "...", "tool": "<name>", "args": {...}}
      {"thought": "...", "finish": {...result...}}
    Returns the parsed dict, or None if parsing failed entirely.
    """
    decision = parse_json(raw, default=None)
    if not isinstance(decision, dict):
        return None

    if "finish" in decision or "tool" in decision:
        return decision

    return None
