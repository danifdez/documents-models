"""Tool registry and decorator for agents."""

from typing import Any, Callable, Dict

from agent.types import ToolContext, ToolSpec

TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def tool(name: str, description: str, args_schema: Dict[str, str], kind: str = "python"):
    """Register a function as an agent tool.

    The function signature must be (args: dict, ctx: ToolContext) -> dict.
    """
    def deco(fn: Callable[[Dict[str, Any], ToolContext], Dict[str, Any]]):
        TOOL_REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            args_schema=args_schema,
            run=fn,
            kind=kind,
        )
        return fn
    return deco
