"""workspace_research: agent that searches/reads workspace content.

Invoked as a tool by the personal assistant. It searches and reads internally
and returns a compact, structured answer — the parent never sees the raw hits
or file bodies, only the summary. That is the parent's context compression.
"""

import os

from lib.llm.prompts import load_prompt
from lib.framework.agent import AgentSpec

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# What this agent must return: the runner appends it to the system prompt (so
# the model knows the exact shape) and parses the reply against it.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Answer to the research question, ≤200 words, leading with "
                "the answer. If nothing relevant was found, say so."
            ),
        },
        "sources": {
            "type": "array",
            "description": (
                "Every workspace item the answer relies on, with the exact "
                "kind and id seen in the tool results. Empty if none."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "notes", "tasks", "projects", "resources", "docs",
                            "canvases", "events", "knowledge", "entities",
                            "datasets",
                        ],
                    },
                    "id": {"type": "integer"},
                    "name": {
                        "type": "string",
                        "description": "Title/name as seen in the results.",
                    },
                },
                "required": ["kind", "id", "name"],
            },
        },
    },
    "required": ["summary", "sources"],
}

# How this agent is offered as a tool to the parent that lists it.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "workspace_research",
        "description": (
            "Look up or read anything in the user's workspace (notes, "
            "calendar events, indexed resources). This is the ONLY way to "
            "search or read workspace content — you have no direct search "
            "tools, so delegate here. A subagent searches and reads "
            "internally and returns a compact summary; you do NOT see the "
            "raw hits or file contents, only the answer. The result is "
            "structured: `summary` plus `sources` — the {kind, id, name} "
            "of every item the answer relies on.\n\n"
            "Use it whenever the answer lives in the workspace: 'what do I "
            "know about X?', 'do I have a note about Y?', 'summarise my "
            "notes on Z', 'find the conditions of contract W'.\n\n"
            "Do NOT use it to enumerate tasks ('what tasks do I have?' → "
            "list_tasks) or projects (list_projects), nor for any create/"
            "update/delete action — it only reads, never modifies.\n\n"
            "Pairs well with: leaf actions you call AFTER the summary "
            "(create_task, create_calendar_event, update_task) using ids "
            "from the `sources` field, matched by name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Research question in natural language. Be "
                        "specific — the subagent has no other context."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

workspace_research = AgentSpec(
    name="workspace_research",
    config_key="workspace-research-agent",
    system_prompt=load_prompt(_PROMPTS_DIR, "workspace_research.md").strip(),
    tool_names=frozenset({
        "search_workspace",
        "get_resource_content",
        "list_notes",
        "list_tasks",
        "list_projects",
    }),
    max_rounds=3,
    output_schema=OUTPUT_SCHEMA,
    tool_schema=TOOL_SCHEMA,
    input_field="query",
    fallback_max_tokens=600,
)
