"""Shared tool repository.

Importing this package registers every leaf tool (each module self-registers on
import) and exposes the registry, the leaf dispatcher and a helper to resolve a
set of tool names into their schemas. Tools are capabilities with no owner: an
agent picks the ones it needs by name (see `core/agents/`).

Import order below is the order an agent sees tools in (when it lists them all)
— keep it stable.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from lib.framework.tool import REGISTRY, Tool, ToolContext, register  # noqa: F401

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


def schemas_for(tool_names) -> List[Dict[str, Any]]:
    """Resolve a set/list of tool names into their JSON schemas, in registry
    (import) order so the model always sees a stable ordering. Names that are
    not leaf tools (e.g. an agent invoked as a tool) are skipped here — the
    caller layers those in."""
    wanted = set(tool_names)
    return [t.schema for name, t in REGISTRY.items() if name in wanted]


def execute_leaf(name: str, args_json: str, ctx: ToolContext) -> Dict[str, Any]:
    """Dispatch a single leaf tool call: parse the JSON args and run the
    registered executor."""
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


def summarize_leaf(name: str, result: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Compact one-line summary + optional entity for the UI tool card. Each
    tool provides its own success summary; anything else falls back to the
    generic error/OK line."""
    if not isinstance(result, dict):
        return "OK", None
    tool = REGISTRY.get(name)
    if tool is not None and tool.summarize is not None:
        out = tool.summarize(result)
        if out is not None:
            return out
    if "error" in result:
        return f"error: {result['error']}", None
    return "OK", None
