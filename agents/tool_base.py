"""Tool abstraction — the framework the leaf tools are instances of.

Defines `Tool`, `ToolContext`, `register` and the `REGISTRY`. It lives in
`core/agents` (with the agent abstraction) rather than in `core/tools` because
it is framework, not a capability. It is deliberately a pure module (no imports
beyond the stdlib) so `core/tools` can import it without dragging in the agent
wiring — see this package's `__init__` for how the cycle is avoided.

Each leaf tool in `core/tools` self-registers a `Tool` on import;
`tools/__init__.py` imports every tool module to populate the registry, then
derives the catalog and the leaf dispatcher from it. Tools know nothing about
tasks or agents: they are just callable capabilities. Who may call each tool is
decided by each agent's declared tool list (see `core/agents/`), not by flags
here.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

# execute: (args, ctx) -> result dict written back to the model as JSON.
ExecuteFn = Callable[[Dict[str, Any], "ToolContext"], Dict[str, Any]]
# summarize: (result) -> (summary_text, entity_or_None), or None to fall back to
# the generic error/OK summary. `entity` feeds the UI's tool card so it can
# render a delete action on things the tool just created.
SummarizeFn = Callable[[Dict[str, Any]], Optional[Tuple[str, Optional[Dict[str, Any]]]]]


@dataclass
class ToolContext:
    """Per-call context a tool executor may need beyond its own args. Built once
    per tool-call round from the job payload. `folder_scope` travels here so
    folder-aware tools can act on the caller's working folder; a tool that
    doesn't need it just ignores it."""
    owner_segment: str = "assistants"
    owner_id: Optional[int] = None
    job_id: Optional[int] = None
    folder_scope: str = ""
    payload: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}


@dataclass
class Tool:
    """A tool as the model sees it (schema) and as the worker runs it (execute).

    A leaf capability: it runs an executor and returns a result dict. It carries
    no visibility flags — each agent declares which tools it can use.
    """
    schema: Dict[str, Any]
    execute: ExecuteFn
    summarize: Optional[SummarizeFn] = None

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


REGISTRY: Dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    """Insert a tool into the registry, preserving import order (which is the
    order the model sees in the catalog)."""
    name = tool.name
    if name in REGISTRY:
        raise ValueError(f"duplicate tool {name!r}")
    REGISTRY[name] = tool
    return tool
