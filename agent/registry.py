"""Loads agent definitions from models/agent/agents/*.json at import time."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from agent.types import AgentDefinition, ModelSpec

logger = logging.getLogger(__name__)

_AGENTS_DIR = Path(__file__).resolve().parent / "agents"
_AGENTS: Dict[str, AgentDefinition] = {}
_LOADED = False


def _build_definition(data: dict) -> AgentDefinition:
    return AgentDefinition(
        name=data["name"],
        system_prompt=data.get("system_prompt", ""),
        tools=list(data.get("tools", [])),
        model=ModelSpec.from_any(data.get("model")),
        max_steps=int(data.get("max_steps", 8)),
        max_depth=int(data.get("max_depth", 2)),
        tool_defaults=dict(data.get("tool_defaults", {})),
        input_schema=dict(data.get("input_schema", {})),
        output_schema=dict(data.get("output_schema", {})),
    )


def _load_all() -> None:
    global _LOADED
    if _LOADED:
        return

    if not _AGENTS_DIR.exists():
        _LOADED = True
        return

    for path in sorted(_AGENTS_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            agent_def = _build_definition(data)
            _AGENTS[agent_def.name] = agent_def
            logger.info("Registered agent: %s", agent_def.name)
        except Exception:
            logger.exception("Failed to load agent definition from %s", path)

    _LOADED = True


def get_agent(name: str) -> Optional[AgentDefinition]:
    _load_all()
    return _AGENTS.get(name)


def all_agents() -> Dict[str, AgentDefinition]:
    _load_all()
    return dict(_AGENTS)


def has_agent(name: str) -> bool:
    _load_all()
    return name in _AGENTS
