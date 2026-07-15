"""Agent abstraction: what an agent IS *and* how it runs.

An `AgentSpec` describes an agent — its prompt, the tools it may use (by name,
resolved against the shared `core/tools` repository), how many tool rounds it
gets, and how it finishes (a free-text reply, or a structured object matching an
output schema). The personal assistant and every subagent are instances of the
same abstraction; the only difference is the fields they set.

It is also the abstraction a task USES: a caller imports one agent and drives it
directly — `agent.run(messages, ctx)` for a top-level turn, `agent.run_as_tool(
args_json, ctx)` when it is invoked as a tool by another agent. No separate
runner function to import and wire.

The spec's *fields* are pure data; importing this module drags in nothing but
stdlib. The *methods* need the engine (`agents.loop`) and the agent repository
(`agents` — for `dispatch_tool`/`AGENTS`), so those are imported lazily, inside
the methods, at call time. That keeps the import graph acyclic (`core/tools`
and `core/agents` depend on this module, never the other way at import time) and
lets the harness monkeypatch `agents.dispatch_tool` and have it take effect.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentSpec:
    """An agent: prompt + declared tools + turn budget + how it finishes.

    - `tool_names`: the tools this agent may call, by name. Leaves live in
      `core/tools`; a name that resolves to another agent makes that agent
      callable-as-a-tool from here.
    - `output_schema`: None → the agent ends with a free-text reply (the caller
      decides what to do with the conversation). A dict → the agent must end
      with a JSON object of that shape; `run` returns it.
    - `emits_tool_events`: push live tool cards to the UI (the user-facing
      assistant does; subagents don't).
    - `tool_schema`: how this agent is offered to a PARENT agent that lists it in
      its own `tool_names`. None for a top-level agent that nobody invokes.
    - `input_field`: when invoked as a tool, the JSON property holding its input.
    - `requires_folder`: this agent is useless without a working folder, so a
      parent hides it from its catalog when the turn carries no `folderScope`
      (offering it anyway just degrades tool selection).
    """
    name: str
    config_key: str
    system_prompt: str
    tool_names: frozenset
    max_rounds: int = 3
    output_schema: Optional[Dict[str, Any]] = None
    emits_tool_events: bool = False
    tool_schema: Optional[Dict[str, Any]] = None
    input_field: str = "query"
    fallback_max_tokens: int = 600
    requires_folder: bool = False

    # ── Behaviour ─────────────────────────────────────────────────────────────

    def tools(self, ctx) -> List[Dict[str, Any]]:
        """The schemas this agent sees: its leaf tools (in registry order)
        followed by the schemas of any agents it may invoke as tools. A
        folder-dependent agent is dropped from the catalog when the turn carries
        no working folder — offering it without a folder only degrades tool
        selection."""
        from tools import schemas_for
        from agents import AGENTS

        leaves = schemas_for(self.tool_names)
        sub_schemas = []
        for name in self.tool_names:
            sub = AGENTS.get(name)
            if sub is None or sub.tool_schema is None:
                continue
            if sub.requires_folder and not ctx.folder_scope:
                continue
            sub_schemas.append(sub.tool_schema)
        return leaves + sub_schemas

    def run(
        self, messages: List[Dict[str, Any]], ctx,
    ) -> Optional[Dict[str, Any]]:
        """Run this agent as a top-level turn over already-built messages (the
        caller owns the system prompt). Returns the structured object for a
        schema agent, or None for a free-reply agent (the caller renders the
        reply from `messages`, which is mutated in place)."""
        from agents import dispatch_tool
        from agents.loop import run_agent_loop

        return run_agent_loop(self, messages, ctx, self.tools(ctx), dispatch_tool)

    def run_as_tool(self, args_json: str, ctx) -> Dict[str, Any]:
        """Run this agent when it is invoked as a tool by another agent: parse
        its input, build its own messages from its system prompt (it never sees
        the parent's history — that is the parent's context compression), and run
        its loop."""
        from agents import dispatch_tool
        from agents.loop import run_agent_loop
        from lib.llm.config import get_task_config

        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            return {"error": "invalid_subagent_args"}
        if not isinstance(args, dict):
            return {"error": "invalid_subagent_args"}
        query = str(args.get(self.input_field) or "").strip()
        if not query:
            return {"error": f"missing_{self.input_field}"}

        now = datetime.now().astimezone().replace(microsecond=0)
        system_content = (
            f"{self.system_prompt}\n\nCurrent date and time: {now.isoformat()} "
            f"({now:%A}). Resolve relative or time-only references like "
            "'at 10pm', 'tomorrow' or 'next Monday' against it."
        )
        if self.output_schema is not None:
            system_content = (
                f"{system_content}\n\nOUTPUT SCHEMA — once you stop calling "
                "tools, reply with a single JSON object matching this JSON "
                "Schema, and nothing else:\n"
                f"{json.dumps(self.output_schema, ensure_ascii=False)}"
            )
        if not bool(get_task_config(self.config_key).get("enable_thinking", False)):
            system_content = f"{system_content}\n\n/no_think"

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        # A subagent crash must not kill the parent turn: retry once, then degrade.
        for attempt in range(2):
            try:
                result = run_agent_loop(
                    self, messages, ctx, self.tools(ctx), dispatch_tool,
                )
                return result if result is not None else {"summary": ""}
            except Exception:
                logger.exception(
                    "agent[%s]: run failed (attempt %d/2)", self.name, attempt + 1,
                )
        return {"error": "subagent_failed", "tool": self.name}
