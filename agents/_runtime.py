"""Agents: the personal assistant and the subagents it delegates to.

Every agent is an `AgentSpec` (see `base.py`) run by the same `run_agent_loop`.
This package wires them together:

- `AGENTS`   — agents invocable as a tool by another agent, keyed by name.
- `run_agent(spec, messages, ctx)` — run a top-level agent over prebuilt messages.
- `dispatch_tool(name, args_json, ctx)` — resolve one tool call: a nested agent
  runs its own loop; anything else is a leaf executed from `core/tools`.

`dispatch_tool` is the single point where a tool call is routed, so tests
intercept here to trace calls and to stub subagents.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.model_config import get_task_config
from tools import ToolContext, execute_leaf, schemas_for

from .base import AgentSpec, run_agent_loop
from .assistant import MAIN_AGENT
from .workspace_research import WORKSPACE_RESEARCH
from .folder_assistant import FOLDER_ASSISTANT

logger = logging.getLogger(__name__)

# Agents that can be invoked as a tool from another agent's tool_names.
AGENTS: Dict[str, AgentSpec] = {
    WORKSPACE_RESEARCH.name: WORKSPACE_RESEARCH,
    FOLDER_ASSISTANT.name: FOLDER_ASSISTANT,
}


def tools_for_agent(spec: AgentSpec, ctx: ToolContext) -> List[Dict[str, Any]]:
    """The schemas this agent sees: its leaf tools (in registry order) followed
    by the schemas of any agents it may invoke as tools. A folder-dependent
    agent is dropped from the catalog when the turn carries no working folder —
    offering it without a folder only degrades tool selection."""
    leaves = schemas_for(spec.tool_names)
    sub_schemas = []
    for name in spec.tool_names:
        sub = AGENTS.get(name)
        if sub is None or sub.tool_schema is None:
            continue
        if sub.requires_folder and not ctx.folder_scope:
            continue
        sub_schemas.append(sub.tool_schema)
    return leaves + sub_schemas


def run_agent(
    spec: AgentSpec, messages: List[Dict[str, Any]], ctx: ToolContext,
) -> Optional[Dict[str, Any]]:
    """Run a top-level agent over already-built messages (the caller owns the
    system prompt). Returns the structured object for a schema agent, or None
    for a free-reply agent (the caller renders the reply from `messages`)."""
    return run_agent_loop(spec, messages, ctx, tools_for_agent(spec, ctx), dispatch_tool)


def run_subagent(spec: AgentSpec, args_json: str, ctx: ToolContext) -> Dict[str, Any]:
    """Run an agent invoked as a tool: parse its input, build its own messages
    from its system prompt (it never sees the parent's history — that is the
    parent's context compression), and run its loop."""
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid_subagent_args"}
    if not isinstance(args, dict):
        return {"error": "invalid_subagent_args"}
    query = str(args.get(spec.input_field) or "").strip()
    if not query:
        return {"error": f"missing_{spec.input_field}"}

    now = datetime.now().astimezone().replace(microsecond=0)
    system_content = (
        f"{spec.system_prompt}\n\nCurrent date and time: {now.isoformat()} "
        f"({now:%A}). Resolve relative or time-only references like "
        "'at 10pm', 'tomorrow' or 'next Monday' against it."
    )
    if spec.output_schema is not None:
        system_content = (
            f"{system_content}\n\nOUTPUT SCHEMA — once you stop calling tools, "
            "reply with a single JSON object matching this JSON Schema, and "
            f"nothing else:\n{json.dumps(spec.output_schema, ensure_ascii=False)}"
        )
    if not bool(get_task_config(spec.config_key).get("enable_thinking", False)):
        system_content = f"{system_content}\n\n/no_think"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]
    # A subagent crash must not kill the parent turn: retry once, then degrade.
    for attempt in range(2):
        try:
            result = run_agent_loop(
                spec, messages, ctx, tools_for_agent(spec, ctx), dispatch_tool,
            )
            return result if result is not None else {"summary": ""}
        except Exception:
            logger.exception(
                "agent[%s]: run failed (attempt %d/2)", spec.name, attempt + 1,
            )
    return {"error": "subagent_failed", "tool": spec.name}


def dispatch_tool(name: str, args_json: str, ctx: ToolContext) -> Dict[str, Any]:
    """Route one tool call: a nested agent runs its own loop; anything else is a
    leaf executed from the shared tool repository."""
    spec = AGENTS.get(name)
    if spec is not None:
        return run_subagent(spec, args_json, ctx)
    return execute_leaf(name, args_json, ctx)
