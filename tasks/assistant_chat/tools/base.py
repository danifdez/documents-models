"""Tool registry for the assistant/agent chat.

Each tool lives in its own module and self-registers a `Tool` on import.
`tools/__init__.py` imports every tool module so the registry is populated,
then derives the catalog (`ASSISTANT_TOOLS`), the visibility subsets and the
dispatcher from it. Nothing here reaches back into `assistant_chat.py`, so the
package imports cleanly with no cycle.
"""

from dataclasses import dataclass, field
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
    per tool-call round from the job payload."""
    kind: str = "assistant"
    owner_segment: str = "assistants"
    owner_id: Optional[int] = None
    job_id: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubagentSpec:
    """Configuration of a subagent the main assistant can invoke as a tool.

    A subagent is NOT a separate job. It runs synchronously inside the tool
    executor of the parent: a fresh chat_with_tools loop, scoped tools, scoped
    system prompt, returns a single text summary to the parent.
    """
    name: str                        # tool name registered in the catalog
    config_key: str                  # tasks.json entry for max_tokens, n_ctx, model
    system_prompt: str               # subagent's own system prompt (English)
    tool_names: frozenset            # subset of the catalog available to it
    input_field: str = "query"       # name of the JSON property holding the input
    max_rounds: int = 3              # internal MAX_TOOL_ROUNDS for this subagent
    fallback_max_tokens: int = 600   # cap used if tasks.json doesn't override
    # JSON Schema of the subagent's final reply. Each subagent defines its own.
    # The runner shows it to the LLM (so it knows exactly what to return) and
    # parses the reply against it — with a grammar-forced retry if needed —
    # so the parent receives structured data, not free text to re-extract.
    output_schema: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    """A tool as the model sees and as the worker runs it.

    `subagent` is set for tools that delegate to a subagent instead of running a
    leaf executor; those carry no `execute` (dispatch is handled by the parent
    loop, which owns the LLM). Visibility flags mirror the former hand-kept sets:
    `agent_allowed` (agents may call it), `folder_scoped` (dead weight without a
    working folder), `hidden_from_main` (reachable only via a subagent).
    """
    schema: Dict[str, Any]
    execute: Optional[ExecuteFn] = None
    summarize: Optional[SummarizeFn] = None
    agent_allowed: bool = False
    folder_scoped: bool = False
    hidden_from_main: bool = False
    subagent: Optional[SubagentSpec] = None

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
