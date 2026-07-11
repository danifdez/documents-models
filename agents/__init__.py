"""Agents: the personal assistant and the subagents it delegates to.

The shared tool abstraction (`Tool`, `ToolContext`, `register`, `REGISTRY`)
lives here too — in `tool_base.py` — because it is framework, not a capability:
the leaf tools in `core/tools` are its instances. It is a pure module (no
imports beyond the stdlib), so `core/tools` can depend on it without pulling in
the agent wiring.

That wiring (`AgentSpec`, `run_agent_loop`, `AGENTS`, `run_agent`,
`dispatch_tool`, …) lives in `_runtime.py` and depends on `core/tools`. Since
`core/tools` imports the abstraction from THIS package, importing the wiring
eagerly here would form a cycle while `tools` is still initialising. So the
abstraction is exposed eagerly and the wiring lazily (PEP 562 `__getattr__`),
resolved on first access — which always happens after `tools` finished loading.

`dispatch_tool` is the single point where a tool call is routed (a nested agent
runs its own loop; anything else is a leaf executed from `core/tools`), so tests
intercept there to trace calls and to stub subagents.
"""

from .tool_base import REGISTRY, Tool, ToolContext, register  # noqa: F401

_LAZY = frozenset({
    "AgentSpec", "run_agent_loop",
    "AGENTS", "MAIN_AGENT", "WORKSPACE_RESEARCH", "FOLDER_ASSISTANT",
    "tools_for_agent", "run_agent", "run_subagent", "dispatch_tool",
})


def __getattr__(name):
    # Deferred to break the tools <-> agents import cycle (see module docstring).
    if name in _LAZY:
        from . import _runtime
        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
