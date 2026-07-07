"""folder_assistant: subagent that operates on the working folder."""

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
                "What happened, ≤120 words: which file, what action, success "
                "or awaiting confirmation."
            ),
        },
        "files": {
            "type": "array",
            "description": (
                "Files involved in the operation, with the exact "
                "indexedFileId seen in the tool results when available."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "indexedFileId": {"type": "integer"},
                    "filename": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": [
                            "found", "read", "created",
                            "overwrite_pending", "delete_pending",
                        ],
                    },
                },
                "required": ["filename", "action"],
            },
        },
        "pendingConfirmation": {
            "type": "boolean",
            "description": "True if a confirmation card is awaiting the user.",
        },
    },
    "required": ["summary", "files", "pendingConfirmation"],
}

FOLDER_ASSISTANT_PROMPT = load_prompt(_PROMPTS_DIR, "folder_assistant.md").strip()


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "folder_assistant",
            "description": (
                "Delegate a task on the assistant's working folder to a "
                "subagent specialised in folder operations. The subagent "
                "combines folder_search, folder_read, folder_write and "
                "folder_delete internally and returns a structured result: "
                "`summary` of what was done, `files` involved (with "
                "indexedFileId/filename/action, usable in follow-up "
                "folder_read/folder_write calls) and `pendingConfirmation` "
                "when a card is awaiting the user.\n\n"
                "When to use it: multi-step folder operations where the "
                "parent would otherwise pull large file contents into its "
                "context. Examples: 'modify the README adding section X', "
                "'create a shopping list using items from yesterday's "
                "notes', 'find the contract and add a clause about Y'.\n\n"
                "When NOT to use it: a single, direct folder action where "
                "the inputs are already in hand. 'Write a file called X "
                "with content Y' (the user provided the content) → "
                "folder_write. 'Delete the file todo.md' → folder_delete. "
                "Subagents add latency; reserve them for search + read + "
                "write/delete combined.\n\n"
                "Pairs well with: workspace_research when the file content "
                "needs facts from notes/tasks (workspace_research first, "
                "then folder_assistant with the facts in the task)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Natural-language description of the folder "
                            "operation. The subagent has no other context — "
                            "be explicit about which file, what change, and "
                            "any content the user provided."
                        ),
                    },
                },
                "required": ["task"],
            },
        },
    },
    subagent=SubagentSpec(
        name="folder_assistant",
        config_key="folder-assistant-agent",
        system_prompt=FOLDER_ASSISTANT_PROMPT,
        tool_names=frozenset({
            "folder_search",
            "folder_read",
            "folder_write",
            "folder_delete",
        }),
        input_field="task",
        max_rounds=3,
        fallback_max_tokens=800,
        output_schema=OUTPUT_SCHEMA,
    ),
    folder_scoped=True,
))
