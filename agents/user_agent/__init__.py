"""The user-created agent — a top-level, user-facing agent scoped to a folder.

Unlike the personal `assistant` (built-in, memory-backed, full workspace
toolset), a user agent is created on the backend with a custom persona
(`systemPrompt`) and a working folder (`folderScope`). It is the abstraction the
`agent-chat` task drives; its system prompt is assembled per turn by that handler
(user persona + generic base + orientation + date + working folder), so
`system_prompt` stays empty here.

Its toolset is a property of how the agent was CREATED, not of the request: an
agent scoped to a working folder is created with the folder operations; one
created without a folder is a purely textual agent (a redactor, a translator…)
with no tools at all. It has NO personal-assistant memory and none of the
task/calendar/notes leaves; those belong to the assistant.
"""

import os

from lib.llm.prompts import load_prompt
from lib.framework.agent import AgentSpec

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Generic context always layered under the user's persona: the folder-tool
# orientation a user agent needs whatever custom `systemPrompt` it was created
# with. It is also the whole persona when the payload carries none.
BASE_SYSTEM_PROMPT = load_prompt(_PROMPTS_DIR, "agent_system.md").strip()

# Multi-tool composition orientation, concatenated into the system prompt. Each
# agent owns its own copy (see `core/agents/`'s per-agent prompt convention).
MULTI_TOOL_ORIENTATION = load_prompt(_PROMPTS_DIR, "multi_tool_orientation.md").strip()

# The folder operations — the tools a user agent is created with WHEN it is
# scoped to a working folder. An agent created without a folder gets none of
# them.
FOLDER_TOOLS = frozenset({
    "folder_search",
    "folder_read",
    "folder_write",
    "folder_delete",
})


def _spec(tool_names) -> AgentSpec:
    return AgentSpec(
        name="user_agent",
        config_key="agent-chat",
        system_prompt="",
        tool_names=tool_names,
        max_rounds=3,
        output_schema=None,      # ends with a free-text reply, streamed by the handler
        emits_tool_events=True,
    )


def user_agent_for(payload) -> AgentSpec:
    """Build the agent spec for one turn with the toolset the agent was CREATED
    with. A user agent's tools are a property of the agent, not of the request:
    an agent scoped to a working folder is created with the folder operations;
    one created without a folder is a purely textual agent (persona only) and
    gets no tools — offering folder tools it can't use only degrades selection."""
    has_folder = bool((payload.get("folderScope") or "").strip())
    return _spec(FOLDER_TOOLS if has_folder else frozenset())


# Default instance: the folder-scoped agent with the full folder toolset. Kept
# for import compatibility and catalog listing; the handler builds the per-agent
# spec via `user_agent_for(payload)`.
user_agent = _spec(FOLDER_TOOLS)
