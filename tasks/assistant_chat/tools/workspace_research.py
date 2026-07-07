"""workspace_research: subagent that searches/reads workspace content."""

import os

from services.prompts import load_prompt

from .base import SubagentSpec, Tool, register

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# What this subagent's LLM must return. The runner appends it to the system
# prompt (so the model knows the exact shape) and parses the reply against it,
# grammar-forcing a retry when the reply doesn't conform.
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

WORKSPACE_RESEARCH_PROMPT = load_prompt(_PROMPTS_DIR, "workspace_research.md").strip()


register(Tool(
    schema={
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
    },
    subagent=SubagentSpec(
        name="workspace_research",
        config_key="workspace-research-agent",
        system_prompt=WORKSPACE_RESEARCH_PROMPT,
        tool_names=frozenset({
            "search_workspace",
            "get_resource_content",
            "list_notes",
            "list_tasks",
            "list_projects",
        }),
        input_field="query",
        max_rounds=3,
        fallback_max_tokens=600,
        output_schema=OUTPUT_SCHEMA,
    ),
))
