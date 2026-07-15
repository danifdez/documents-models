"""The personal assistant — the top-level, user-facing agent.

It declares its tool set here: the leaf actions over tasks, calendar and notes,
plus the two agents it can delegate to (workspace_research, folder_assistant).
Its system prompt is assembled per turn by the task handler (persona + memory +
date + working folder), so `system_prompt` stays empty here — the handler feeds
`assistant.run(messages, ctx)` the built messages. The prompt building blocks
the handler assembles live with the agent, exposed here:

- `DEFAULT_SYSTEM_PROMPT`   — the default persona, used when the payload carries
                              no custom `systemPrompt`.
- `MULTI_TOOL_ORIENTATION`  — multi-tool composition orientation, concatenated
                              into the system prompt.
"""

import os

from lib.llm.prompts import load_prompt
from lib.framework.agent import AgentSpec

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

DEFAULT_SYSTEM_PROMPT = load_prompt(_PROMPTS_DIR, "assistant_system.md").strip()

# Multi-tool composition orientation, injected into the system prompt: compose,
# chain, don't repeat, stop when done.
MULTI_TOOL_ORIENTATION = load_prompt(_PROMPTS_DIR, "multi_tool_orientation.md").strip()

assistant = AgentSpec(
    name="assistant",
    config_key="assistant-chat",
    system_prompt="",
    tool_names=frozenset({
        # Leaf actions.
        "create_note",
        "create_task",
        "list_projects",
        "list_tasks",
        "update_task",
        "delete_task",
        "set_task_reminder",
        "clear_task_reminder",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
        "mark_event_occurrence_done",
        # Delegated agents (invoked as tools).
        "workspace_research",
        "folder_assistant",
    }),
    max_rounds=3,
    output_schema=None,          # ends with a free-text reply, streamed by the handler
    emits_tool_events=True,
)
