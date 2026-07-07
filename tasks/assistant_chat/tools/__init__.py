"""Assistant/agent tool catalog.

Importing this package registers every tool (each module self-registers on
import) and exposes the derived catalog, the visibility subsets and the leaf
dispatcher. The parent chat loop (`assistant_chat.py`) owns subagent dispatch
and the LLM, so it stays there; everything else lives here.

Import order below is the order the model sees tools in — keep it stable.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .base import REGISTRY, SubagentSpec, Tool, ToolContext, register  # noqa: F401

# Leaf tools, in catalog order.
from . import search_workspace  # noqa: F401,E402
from . import create_note  # noqa: F401,E402
from . import create_task  # noqa: F401,E402
from . import get_resource_content  # noqa: F401,E402
from . import list_projects  # noqa: F401,E402
from . import list_notes  # noqa: F401,E402
from . import folder_delete  # noqa: F401,E402
from . import folder_search  # noqa: F401,E402
from . import folder_read  # noqa: F401,E402
from . import folder_write  # noqa: F401,E402
from . import list_tasks  # noqa: F401,E402
from . import update_task  # noqa: F401,E402
from . import delete_task  # noqa: F401,E402
from . import set_task_reminder  # noqa: F401,E402
from . import clear_task_reminder  # noqa: F401,E402
from . import create_calendar_event  # noqa: F401,E402
from . import update_calendar_event  # noqa: F401,E402
from . import delete_calendar_event  # noqa: F401,E402
from . import mark_event_occurrence_done  # noqa: F401,E402
# Subagent tools last (they only carry schema + SubagentSpec + prompt).
from . import workspace_research  # noqa: F401,E402
from . import folder_assistant  # noqa: F401,E402


# The catalog the model sees, in registration (import) order.
ASSISTANT_TOOLS: List[Dict[str, Any]] = [t.schema for t in REGISTRY.values()]

# Tools an agent is allowed to call. Strictly limited to its own working
# folder; agents never learn the other tools exist.
AGENT_ALLOWED_TOOLS = frozenset(t.name for t in REGISTRY.values() if t.agent_allowed)

# folder_* leaves + the folder subagent: dead weight without a working folder.
FOLDER_SCOPED_TOOLS = frozenset(t.name for t in REGISTRY.values() if t.folder_scoped)

# Workspace content search/read: hidden from the main assistant so the only path
# to look up or read workspace content is the workspace_research subagent.
SEARCH_SUBAGENT_TOOLS = frozenset(t.name for t in REGISTRY.values() if t.hidden_from_main)

# Tools that delegate to a subagent, keyed by tool name.
SUBAGENT_SPECS: Dict[str, SubagentSpec] = {
    t.name: t.subagent for t in REGISTRY.values() if t.subagent is not None
}


def tools_for_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tools the model sees. Agents get only the folder_* leaves. The personal
    assistant gets the catalog minus the search/read leaves (reachable only via
    workspace_research) and minus folder tools when it has no working folder."""
    if (payload.get("kind") or "assistant") == "agent":
        return [t.schema for t in REGISTRY.values() if t.agent_allowed]
    hidden = set(SEARCH_SUBAGENT_TOOLS)
    if not (payload.get("folderScope") or "").strip():
        hidden |= FOLDER_SCOPED_TOOLS
    return [t.schema for t in REGISTRY.values() if t.name not in hidden]


def execute_leaf(name: str, args_json: str, ctx: ToolContext) -> Dict[str, Any]:
    """Dispatch a single non-subagent tool call. Parses the JSON args and runs
    the registered executor. Subagent dispatch and the agent allowlist guard are
    owned by the parent loop, which calls this only for leaf tools."""
    tool = REGISTRY.get(name)
    if tool is None or tool.execute is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return tool.execute(args, ctx)


def summarize(name: str, result: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Compact one-line summary + optional entity for the UI tool card. Mirrors
    the former central if/elif: subagents summarise from their returned text,
    each tool provides its own success summary, and anything else falls back to
    the generic error/OK line."""
    if not isinstance(result, dict):
        return "OK", None
    if name in SUBAGENT_SPECS and isinstance(result.get("summary"), str):
        text = result["summary"]
        return text[:200] + ("…" if len(text) > 200 else ""), None
    tool = REGISTRY.get(name)
    if tool is not None and tool.summarize is not None:
        out = tool.summarize(result)
        if out is not None:
            return out
    if "error" in result:
        return f"error: {result['error']}", None
    return "OK", None
