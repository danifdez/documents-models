"""Agents: the personal assistant and the subagents it delegates to.

Each agent lives in its own folder under this package — its spec, its schemas
and its prompts together:

- `assistant/`           — the top-level, built-in personal assistant.
- `user_agent/`          — the top-level, user-created folder agent.
- `workspace_research/`  — subagent: reads/searches workspace content.
- `folder_assistant/`    — subagent: operates on the user's working folder.
- `memory_agent/`        — persistent-memory mini-agent: the handler runs it
                           after every assistant turn (not an `AgentSpec`, not
                           invocable as a tool — it never enters `AGENTS`).
- `loop.py`              — the shared engine (`run_agent_loop`) that runs any
                           agent's tool-call rounds.

An agent IS the abstraction a task uses: import it and drive it directly —
`assistant.run(messages, ctx)`. The behaviour (run / run_as_tool / tools) lives
on `AgentSpec` in `lib.framework.agent`; this package only wires the concrete
agents together and owns the tool-dispatch routing.

- `AGENTS`   — agents invocable as a tool by another agent, keyed by name.
- `dispatch_tool(name, args_json, ctx)` — resolve one tool call: a nested agent
  runs its own loop; anything else is a leaf executed from `core/tools`.

`core/tools` depends only on `lib.framework.tool`, never on this package, so the
import graph is acyclic and everything is wired eagerly here.

`dispatch_tool` is the single point where a tool call is routed, so tests
intercept here to trace calls and to stub subagents.
"""

import logging
from typing import Any, Dict

from lib.framework.agent import AgentSpec
from tools import ToolContext, execute_leaf

from .loop import run_agent_loop
from .assistant import assistant
from .user_agent import user_agent
from .workspace_research import workspace_research
from .folder_assistant import folder_assistant

logger = logging.getLogger(__name__)

__all__ = [
    "AgentSpec", "run_agent_loop", "AGENTS", "dispatch_tool",
    "assistant", "user_agent", "workspace_research", "folder_assistant",
]

# Agents that can be invoked as a tool from another agent's tool_names.
AGENTS: Dict[str, AgentSpec] = {
    workspace_research.name: workspace_research,
    folder_assistant.name: folder_assistant,
}


def dispatch_tool(name: str, args_json: str, ctx: ToolContext) -> Dict[str, Any]:
    """Route one tool call: a nested agent runs its own loop; anything else is a
    leaf executed from the shared tool repository."""
    spec = AGENTS.get(name)
    if spec is not None:
        return spec.run_as_tool(args_json, ctx)
    return execute_leaf(name, args_json, ctx)
