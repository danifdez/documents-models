"""Parses an agent's LLM decision into a structured action."""

import logging

from lib.framework.agent_protocol import ModelOutcome
from lib.llm.json import parse_json

logger = logging.getLogger(__name__)


def parse_decision(raw: str) -> ModelOutcome:
    """
    Expects the model to emit JSON of one of these shapes:
      {"thought": "...", "tool": "<name>", "args": {...}}
      {"thought": "...", "finish": {...result...}}
    Returns the canonical model outcome used by every agent runner.
    """
    decision = parse_json(raw, default=None)
    return ModelOutcome.from_structured_decision(decision)
