"""The personal assistant — the top-level, user-facing agent.

It declares its tool set here: the leaf actions over tasks, calendar and notes,
plus the two agents it can delegate to (workspace_research, folder_assistant).
Its system prompt is assembled per turn by the task handler (persona + memory +
date + working folder), so `system_prompt` stays empty here — the handler feeds
`run_agent` the built messages.
"""

from .base import AgentSpec

MAIN_AGENT = AgentSpec(
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
