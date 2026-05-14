"""
Pseudo-tool 'finish'. The loop driver intercepts decisions with a 'finish'
key directly, so this tool exists only so it shows up in the prompt catalog.
"""

from agent.tools.base import tool
from agent.types import ToolContext


@tool(
    name="finish",
    description="Stop the agent and return the final result. Use when you have a verified answer.",
    args_schema={"result": "object matching the agent's output_schema"},
)
def finish_tool(args, ctx: ToolContext):
    return {"_unreachable": True}
