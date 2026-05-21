"""Personal assistant and agent handler.

Multi-turn chat with system prompt and history. This is NOT a Q&A over a file
(that's `ask`). The thread is persisted on the NestJS backend.

Expected payload (common fields):
  {
    "kind": "assistant" | "agent",        # default "assistant" for back-compat
    "ownerType": "main-assistant" | "agent",
    "ownerId": int,                       # id in owner's table
    "name": str,                          # owner's display name
    "systemPrompt": str | null,           # owner's custom prompt; null => default
    "folderScope": str | null,
    "conversation": [{"role": ..., "content": ...}, ...]
  }

For kind="assistant" additionally:
  "assistantId": int,                     # legacy alias of ownerId
  "assistantName": str,                   # legacy alias of name
  "assistantSystem": bool,                # true for the personal assistant
  "memorySnippets": [...],                # injected memory
  "extractMemory": bool                   # run memory extraction

For kind="agent" additionally:
  "agentId": int                          # legacy alias of ownerId

Returns:
  {"reply": str, "memoryAction"?: {...}}  or  {"error": str}
"""

import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import urllib.error

from tasks.assistant_chat.file_writers import (
    ConversionError,
    UnsupportedExtension,
    normalize_and_categorize,
    to_bytes as _writer_to_bytes,
)
import urllib.parse
import urllib.request

from services.llm_service import get_llm_service
from services.model_config import get_llm_params, get_task_config
from utils.job_registry import job_handler

logger = logging.getLogger(__name__)

# Backend HTTP endpoint used both for streaming chunks back to the UI and for
# executing assistant tools (search, etc.). Local-only by design.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")

# How many rounds of tool calls we allow per user turn before forcing a final
# text response. Prevents the model from looping on tools forever.
MAX_TOOL_ROUNDS = 3

# Tools the assistant can call. Schema follows the OpenAI/llama-cpp tools
# convention. Each one is dispatched in `_execute_tool` below.
ASSISTANT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_workspace",
            "description": (
                "Search content in the user's workspace (notes, files, "
                "tasks, knowledge base, canvases). Use it when the "
                "user asks about something that might be saved or when "
                "you need context about existing projects/notes/files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms in natural language.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_note",
            "description": (
                "Create a note in the workspace. Use it when the user "
                "explicitly asks to jot down/note something as a note (e.g. "
                "'jot down a note about X', 'create a note titled Y'). DO NOT use "
                "for pending tasks — for that use create_task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title for the note (max ~80 chars).",
                    },
                    "body": {
                        "type": "string",
                        "description": "Note body. May use markdown.",
                    },
                    "projectId": {
                        "type": "integer",
                        "description": "ID of the project to save it under. Omit for a general note.",
                    },
                },
                "required": ["title", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": (
                "Create a pending task. Use it when the user asks you to "
                "remind them to do something or to jot down a task (e.g. "
                "'remind me about X', 'add task Y', 'I have to do Z')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "What needs to be done, in concise imperative form.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Additional details if the user provided them. Optional.",
                    },
                    "projectId": {
                        "type": "integer",
                        "description": "ID of the project the task belongs to. Omit if general.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_resource_content",
            "description": (
                "Read the full content of a file/document in the "
                "workspace. Combine it with search_workspace: first you search, "
                "obtain the resource id, then read it with this tool if "
                "you need to know its content in order to answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resourceId": {
                        "type": "integer",
                        "description": "Resource ID (returned by search_workspace when collection='resources').",
                    },
                },
                "required": ["resourceId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": (
                "List the user's projects with their id and name. Useful "
                "when the user asks 'what projects do I have?' or as a "
                "preliminary step to associate a note/task with a project."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": (
                "List the user's notes. Useful when the user asks "
                "'what notes do I have?' or 'show me my notes'. If filtering by "
                "project, pass projectId."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "projectId": {
                        "type": "integer",
                        "description": "Project ID. Omit to list all.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "folder_delete",
            "description": (
                "Delete a file from the assistant's working folder. Identify "
                "the file by indexedFileId (preferred) or filename. This tool "
                "ALWAYS shows a confirmation card to the user before the file "
                "is actually removed: never delete without explicit user "
                "approval in the card. If the filename is ambiguous you'll "
                "receive a list of candidates and you must ask the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indexedFileId": {
                        "type": "integer",
                        "description": "Numeric id of the file (preferred).",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Relative filename, e.g. 'shopping-list.md'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "folder_search",
            "description": (
                "Semantic search across the files in the assistant's working "
                "folder. Use it when the user describes a file by content or "
                "topic and does not remember the exact filename. Returns hits "
                "with indexedFileId, filename and a text snippet — to read a "
                "full result, chain with folder_read using its indexedFileId. "
                "DO NOT confuse with search_workspace, which searches notes / "
                "tasks / canvases in the workspace at large — different "
                "collections. If the user mentions 'my files', 'in my folder' "
                "or anything tied to the working folder, use folder_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query describing what to find.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max hits to return. Default 10.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "folder_read",
            "description": (
                "Read a file from the assistant's working folder. Use it when "
                "the user asks to summarise, modify or look at an existing "
                "file. Identify the file by indexedFileId (preferred, when you "
                "saw it in a previous folder_search or folder_write call) or "
                "by filename. If the filename is ambiguous you'll get a list "
                "of candidates — ask the user which one. For non-text files "
                "(PDF, etc.) you receive the extracted text rather than the "
                "binary; the derivedFromExtraction flag is set so you know."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indexedFileId": {
                        "type": "integer",
                        "description": "Numeric id of the file (preferred).",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Relative filename, e.g. 'shopping-list.md'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "folder_write",
            "description": (
                "Create or overwrite a file in the assistant's working folder. "
                "The content you must emit depends on the filename's extension:\n"
                "\n"
                "  TEXT (.md, .markdown, .txt, .csv, .tsv, .json, .xml, .yaml, "
                ".yml, .toml, .ini, .log, .html, .htm, .svg, .py, .js, .ts, "
                ".tsx, .jsx, .sh, .bash, .sql, .css, .scss, .less, .go, .rs, "
                ".rb, .java, .c, .cpp, .h, .hpp, .cs, .php, .r, .kt, .swift, "
                ".dockerfile)\n"
                "    -> `content` is written verbatim as UTF-8.\n"
                "\n"
                "  DOCUMENT (.pdf, .docx, .odt)\n"
                "    -> `content` MUST be MARKDOWN source. The worker renders "
                "it to the target format with pandoc. Supports headings, "
                "lists, tables, code blocks, links, blockquotes. Embedded "
                "images are NOT supported in this version.\n"
                "\n"
                "  SPREADSHEET (.xlsx)\n"
                "    -> `content` MUST be CSV (comma-separated; first row = "
                "column headers). The worker converts via openpyxl. Numbers "
                "and floats are auto-detected; leading-zero strings stay as "
                "strings (phone numbers, zip codes are safe).\n"
                "\n"
                "If `filename` has no extension, `.md` is assumed.\n"
                "\n"
                "If the file already exists and overwrite=false you'll receive "
                "`file_exists` and must ask the user before retrying with "
                "overwrite=true. With overwrite=true on an existing file the "
                "user sees a confirmation card; the change is NOT applied "
                "until they confirm. Unknown extensions return "
                "`unsupported_extension`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Relative filename including extension, e.g. "
                            "'shopping-list.md', 'report.pdf', 'data.xlsx', "
                            "'script.py'. Subfolders allowed (e.g. "
                            "'notes/2025-01-15.md')."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Content shaped according to the extension (see "
                            "tool description): UTF-8 text, markdown source "
                            "for .pdf/.docx/.odt, or CSV for .xlsx."
                        ),
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "Set to true ONLY after the user has explicitly "
                            "agreed to overwrite an existing file. Default false."
                        ),
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": (
                "List the user's pending tasks. Use it when the "
                "user asks 'what tasks do I have?', 'what's pending', "
                "etc. If filtering by project, pass projectId."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "projectId": {
                        "type": "integer",
                        "description": "Project ID. Omit to list all.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "completed"],
                        "description": "Filter by status. By default all are returned.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task",
            "description": (
                "Update an existing task. Identify it by taskId (preferred, when "
                "you saw it in a previous list_tasks call) or by titleQuery "
                "(approximate title match). Send only the fields that change. "
                "To mark a task as done set status='completed'. To re-open a "
                "completed task set status='pending'. To clear the description "
                "or project send null. If titleQuery is ambiguous you'll receive "
                "candidates — ask the user which one and retry with taskId."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "taskId": {"type": "integer"},
                    "titleQuery": {
                        "type": "string",
                        "description": "Approximate title match if taskId unknown.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "completed"],
                    },
                    "title": {"type": "string"},
                    "description": {"type": ["string", "null"]},
                    "projectId": {"type": ["integer", "null"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task",
            "description": (
                "Delete a task entirely. Identify it by taskId (preferred) or "
                "titleQuery (approximate title match). This tool ALWAYS shows a "
                "confirmation card to the user before the task is actually "
                "deleted: never delete without explicit user approval in the "
                "card. If titleQuery is ambiguous you'll receive a list of "
                "candidates — ask the user which one. Use update_task with "
                "status='completed' instead if the user just wants to mark it "
                "as done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "taskId": {"type": "integer"},
                    "titleQuery": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": (
                "Create a calendar event (one-shot or recurring), optionally with an alarm. "
                "Use it when the user describes something with a date/time (an appointment, a "
                "recurring reminder, a one-off alert). Anything with a date goes here — "
                "tasks without a date go to create_task.\n\n"
                "RRULE examples (RFC 5545):\n"
                "  'every day for 7 days at 10pm' -> FREQ=DAILY;COUNT=7;BYHOUR=22;BYMINUTE=0\n"
                "  'every 3 days' -> FREQ=DAILY;INTERVAL=3\n"
                "  'every monday' -> FREQ=WEEKLY;BYDAY=MO\n"
                "  'first friday of every month' -> FREQ=MONTHLY;BYDAY=1FR\n"
                "  'every year on may 18' -> FREQ=YEARLY;BYMONTH=5;BYMONTHDAY=18\n"
                "Omit recurrenceRule for one-shot events. Include alarm only when the user "
                "asks to be reminded/alerted/notified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short, concrete. Stored as-is — not translated.",
                    },
                    "startAt": {
                        "type": "string",
                        "description": "ISO 8601 with TZ offset, e.g. 2026-05-20T22:00:00+02:00.",
                    },
                    "endAt": {
                        "type": "string",
                        "description": "ISO 8601. Omit if it's a point in time, not a span.",
                    },
                    "recurrenceRule": {
                        "type": "string",
                        "description": "Valid RRULE without 'RRULE:' prefix. Omit for one-shot.",
                    },
                    "alarm": {
                        "type": "object",
                        "description": "Optional reminder. Omit if the user did not ask to be alerted.",
                        "properties": {
                            "offsetMinutes": {
                                "type": "integer",
                                "description": "0=at start; negative=before. Default 0 if alarm is present without explicit offset.",
                            },
                            "label": {"type": "string"},
                        },
                        "required": ["offsetMinutes"],
                    },
                    "projectId": {
                        "type": "integer",
                        "description": "Project id; omit if general.",
                    },
                },
                "required": ["title", "startAt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": (
                "Update an existing calendar event. Identify it by eventId (preferred) "
                "or by 'match' (approximate title). Any field is optional; send only "
                "those that change. To clear a field send null (alarm: null to remove "
                "the alarm; recurrenceRule: null to stop the recurrence). Cannot edit "
                "a single occurrence — only the whole event."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "eventId": {"type": "integer"},
                    "match": {
                        "type": "string",
                        "description": "Approximate title match if eventId unknown.",
                    },
                    "title": {"type": "string"},
                    "startAt": {"type": "string"},
                    "endAt": {"type": "string"},
                    "recurrenceRule": {"type": ["string", "null"]},
                    "alarm": {"type": ["object", "null"]},
                    "projectId": {"type": ["integer", "null"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": (
                "Delete a calendar event entirely (stops all future occurrences and "
                "alarms). Identify by eventId (preferred) or 'match' (approximate "
                "title). To skip a single occurrence, edit the event instead — there "
                "is no per-occurrence override."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "eventId": {"type": "integer"},
                    "match": {"type": "string"},
                },
            },
        },
    },
]
# Tools an agent is allowed to call. Strictly limited to its own working
# folder (Cambio #5 / #6). The personal assistant gets the full ASSISTANT_TOOLS
# list; agents only see this subset, both at the prompt level (the model never
# learns the other tools exist) and at the dispatcher level (defensive reject
# if the model hallucinates a name).
AGENT_ALLOWED_TOOLS = {
    "folder_search",
    "folder_read",
    "folder_write",
    "folder_delete",
}


def _tools_for_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the tools the model sees, filtered by the payload's kind."""
    if (payload.get("kind") or "assistant") == "agent":
        return [
            t for t in ASSISTANT_TOOLS
            if t.get("function", {}).get("name") in AGENT_ALLOWED_TOOLS
        ]
    return ASSISTANT_TOOLS


def _owner_id(payload: Dict[str, Any]) -> Optional[int]:
    """Resolve the owner id for streaming / tool-event POSTs.
    Prefers `ownerId`; falls back to legacy `assistantId` or `agentId`."""
    for key in ("ownerId", "assistantId", "agentId"):
        v = payload.get(key)
        if isinstance(v, int):
            return v
    return None


def _owner_type(payload: Dict[str, Any]) -> str:
    """'main-assistant' or 'agent'. Defaults to legacy assistant."""
    t = payload.get("ownerType")
    if t in ("main-assistant", "agent"):
        return t
    return "agent" if (payload.get("kind") == "agent") else "main-assistant"


def _backend_owner_segment(payload: Dict[str, Any]) -> str:
    """`/assistants/:id` or `/agents/:id` prefix for indexed-files endpoints."""
    return "agents" if _owner_type(payload) == "agent" else "assistants"


DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are an agent created by the user for a specific task. "
    "Your scope is strictly limited to the files inside your working folder, "
    "if one is configured. You do not have access to the user's workspace, "
    "notes, memory, or any other resource. If asked about anything beyond your "
    "folder, say you cannot see it and suggest the user redirect the request "
    "to their main assistant."
)


# Flush a partial chunk to the backend roughly every N ms, regardless of how
# many tokens have accumulated. Tuned so the user sees forward motion without
# drowning the HTTP loop in tiny requests.
STREAM_FLUSH_INTERVAL_MS = 120


def _post_stream_chunk(
    owner_segment: str,
    owner_id: int,
    job_id: int,
    chunk: str,
    done: bool = False,
) -> None:
    """Best-effort POST of a partial reply chunk back to the backend. Failures
    are logged but never raised — streaming is purely a UX nicety; the final
    reply still arrives through the normal job-result path.

    When `done=True`, signals that the model has finished generating even if
    the job hasn't fully completed yet (memory extraction may still be
    running). The UI uses this to stop the live caret immediately instead of
    waiting for the final assistantResponse event."""
    if not chunk and not done:
        return
    url = f"{BACKEND_URL}/{owner_segment}/{owner_id}/stream-chunk"
    body = {"jobId": job_id, "chunk": chunk}
    if done:
        body["done"] = True
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2):
            pass
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning("assistant-chat: stream-chunk POST failed: %s", e)

# Valid types for a memory entry. Mirrors the backend enum.
# Three canonical categories (semantic / episodic / procedural in cognitive
# science): `fact` covers stable knowledge about the user or the world
# (includes preferences, data, identity, tools, links); `event` are
# episodes with a concrete moment in time; `instruction` is how the assistant should behave.
MEMORY_TYPES = {"fact", "event", "instruction"}

# Structured extraction/management prompt. Called on every turn of the
# personal assistant. The LLM decides among three actions — `save` (save a
# new memory), `forget` (forget an existing one by id) or `none` (nothing
# to do) — and returns ONE JSON with that decision.
EXTRACT_MEMORY_PROMPT = """You are a memory manager. Read the user's message and the existing memory, and return ONE single JSON deciding what to do.

EXACT schema:
{{
  "action": "save | forget | none",
  "name": "short title if action=save (3-8 words), otherwise \\"\\"",
  "type": "fact | event | instruction if action=save, otherwise \\"fact\\"",
  "body": "the fact to remember if action=save, otherwise \\"\\"",
  "forget_id": <number> if action=forget; null otherwise
}}

Decide the `action`:

- save: the message contains NEW information worth persisting and NOT
  already covered by existing memory. Fill in name/type/body.

- forget: the user explicitly asks to forget/delete/remove a memory
  ("forget that X", "I no longer live in Y", "delete the dentist thing", "forget that I
  like coffee", etc.). Identify the most relevant existing entry from the
  list and return its `id` in `forget_id`. If there's no clear match, use `none`.

- none: greetings, questions, chit-chat, or anything whose information ALREADY
  exists in the existing memory (don't duplicate). Also use `none` if the
  user is only asking something without providing new info.

NEVER save (always `none`):
- Requests to be reminded/notified/alerted about something with a date or time
  ("remind me to take the pill at 10pm", "notify me every Monday to check email",
  "alert me 30 min before the meeting"). Those belong to the calendar, not memory.
- One-off appointments or events stated as future obligations ("meeting Thursday
  at 14h", "dinner with Marta tomorrow"). Those also belong to the calendar.
- Anything the user said to be done (an action to perform), even if no time is
  given — that's a task, not a memory.

How to decide the `type` when action=save:

- event: episodes already past or rare landmark moments worth remembering as
  context (a birth, a move, a marriage). NOT future appointments or recurring
  reminders — those go to the calendar.
  e.g.: "yesterday I signed the contract", "I got married in 2018".

- instruction: STABLE preferences about how the assistant should speak/format/
  behave across all interactions. NOT a one-off "do X for me" request and NOT a
  request to be reminded about something.
  e.g.: "always answer me in Spanish", "no bullet points", "use formal tone".

- fact: EVERYTHING ELSE — stable knowledge about the user or their environment.
  Includes personal data, where they live, relationships, tastes, tools, links.
  e.g.: "I live in Barcelona", "I don't like coffee", "my GitHub is github.com/x".

Rules:
- Do not invent anything not in the message.
- If the fact is already in existing memory (even worded differently), use `none`.
- Do not add text outside the JSON.

EXISTING MEMORY (may be empty):
{memory_list}

USER MESSAGE:
\"\"\"
{message}
\"\"\""""


def _last_user_message(payload: Dict[str, Any]) -> Optional[str]:
    conversation = payload.get("conversation") or []
    for m in reversed(conversation):
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            return str(m["content"]).strip()
    return None


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Accepts either clean JSON or a JSON surrounded by prose. Returns None
    if no valid object can be extracted."""
    if not text:
        return None
    cleaned = _strip_thinking(text)
    # Direct attempt
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Look for the first brace-block within the text
    m = _JSON_OBJECT_RE.search(cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _format_memory_for_prompt(snippets: List[Dict[str, Any]]) -> str:
    """Render existing memory as a numbered list with ids — the LLM uses these
    ids to point at entries to forget, and the dedup check compares against
    the bodies shown here."""
    lines: List[str] = []
    for s in snippets:
        if not isinstance(s, dict):
            continue
        mid = s.get("id")
        name = (s.get("name") or "").strip()
        type_ = (s.get("type") or "other").strip()
        body = (s.get("body") or "").strip()
        if mid is None or not name or not body:
            continue
        lines.append(f"- id={mid} ({type_}) {name}: {body}")
    return "\n".join(lines) if lines else "(no memory yet)"


def _extract_memory_action(
    llm,
    user_message: str,
    memory_snippets: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Second LLM call: decide save/forget/none based on the user message and
    existing memory. Returns a dict shaped like::

        {"action": "save", "save": {name, type, body}}
        {"action": "forget", "forget_id": int}
        None  # for action=none, parse errors, or invalid output

    The caller (backend processor) reacts to the action accordingly. Memory
    ids in `memory_snippets` are the authoritative ones from the DB — the
    forget_id returned by the LLM must match one of them.
    """
    prompt = EXTRACT_MEMORY_PROMPT.format(
        message=user_message,
        memory_list=_format_memory_for_prompt(memory_snippets),
    )
    # For extraction we keep thinking enabled by default: classifying
    # between {save, forget, none} and detecting duplicates requires reasoning.
    # Configurable via `memory_extract_thinking`.
    extract_thinking = bool(cfg.get("memory_extract_thinking", True))
    messages: List[Dict[str, str]] = []
    if not extract_thinking:
        messages.append({"role": "system", "content": "/no_think"})
    messages.append({"role": "user", "content": prompt})
    try:
        raw = llm.chat(messages, max_tokens=int(cfg.get("memory_extract_max_tokens", 600))) or ""
    except Exception:
        logger.exception("assistant-chat: memory extraction failed")
        return None

    obj = _parse_json_object(raw)
    if not obj:
        logger.info("assistant-chat: could not parse memory JSON: %r", raw[:200])
        return None

    action = str(obj.get("action") or "none").strip().lower()

    if action == "save":
        name = str(obj.get("name") or "").strip()
        body = str(obj.get("body") or "").strip()
        type_ = str(obj.get("type") or "fact").strip().lower()
        if type_ not in MEMORY_TYPES:
            type_ = "fact"
        if not name or not body:
            return None
        return {"action": "save", "save": {"name": name[:120], "type": type_, "body": body}}

    if action == "forget":
        try:
            forget_id = int(obj.get("forget_id"))
        except (TypeError, ValueError):
            return None
        # Validate against the snippets we sent — never trust the LLM to
        # invent ids it didn't see.
        known_ids = {s.get("id") for s in memory_snippets if isinstance(s, dict)}
        if forget_id not in known_ids:
            logger.info("assistant-chat: forget_id %r is not in known memory", forget_id)
            return None
        return {"action": "forget", "forget_id": forget_id}

    return None

# How many turns of history we keep as context. The backend persists
# the full thread, but it makes no sense to pass it entirely to the LLM on every
# call — it falls out of context and latency spikes.
DEFAULT_HISTORY_TURNS = 16

# Thinking models (Qwen3, DeepSeek-R1, etc.) emit their reasoning chain
# inside <think>...</think>. The user only wants to see the final response.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks. Also drops an unclosed leading <think>
    block (happens if max_tokens cuts the reasoning off mid-stream)."""
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # If a <think> remains, it never closed — drop everything from it onwards,
    # but only if we still have content before it.
    if "<think>" in cleaned.lower():
        head = _UNCLOSED_THINK_RE.sub("", cleaned)
        cleaned = head if head.strip() else cleaned
    return cleaned.strip()


def _format_memory_block(snippets: List[Dict[str, Any]]) -> str:
    """Format injected memory entries as a single context block.

    Returns an empty string if there is nothing to inject. Caller is expected
    to embed the result into the main system prompt with explicit framing so
    the model actually uses it (a separate system message often gets ignored).
    """
    lines: List[str] = []
    for s in snippets:
        if not isinstance(s, dict):
            continue
        name = (s.get("name") or "").strip()
        type_ = (s.get("type") or "other").strip()
        body = (s.get("body") or "").strip()
        if not name or not body:
            continue
        lines.append(f"- ({type_}) {name}: {body}")
    return "\n".join(lines)


def _build_messages(payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    kind = (payload.get("kind") or "assistant")
    raw_system_prompt = (payload.get("systemPrompt") or "").strip()
    # Agent without custom prompt → use the default scope-restriction prompt.
    if kind == "agent" and not raw_system_prompt:
        system_prompt = DEFAULT_AGENT_SYSTEM_PROMPT
    else:
        system_prompt = raw_system_prompt
    folder_scope = (payload.get("folderScope") or "").strip()
    memory_snippets = payload.get("memorySnippets") or [] if kind == "assistant" else []
    conversation = payload.get("conversation") or []

    # Persistent user memory only applies to the personal assistant. Agents
    # don't have memory; skip the injection entirely (and the prompt section
    # below) when kind != 'assistant'.
    memory_block = _format_memory_block(memory_snippets)
    if memory_block:
        memory_section = (
            "What you already know about the user (persistent memory from "
            "previous conversations). Lean on these facts when they are "
            "relevant to your answer; if the user asks something whose "
            "answer is here, use it directly:\n"
            f"{memory_block}"
        )
        system_prompt = (
            f"{system_prompt}\n\n{memory_section}" if system_prompt else memory_section
        )

    # Qwen3 emits <think>...</think> before every response by default.
    # That consumes tokens (high latency). For conversational chat it adds
    # no value; we disable it via the official `/no_think` flag in the system
    # prompt. Configurable in case we want explicit thinking in another phase.
    enable_thinking = bool(cfg.get("enable_thinking", False))
    if not enable_thinking:
        marker = "/no_think"
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{marker}"
        else:
            system_prompt = marker

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # If the helper has an associated folder, we keep it as a separate
    # context block. No tools use it yet, but the model can
    # mention it when asked.
    if folder_scope:
        messages.append({
            "role": "system",
            "content": f"[WORKING FOLDER]\nPath: {folder_scope}",
        })

    history_turns = int(cfg.get("history_turns", DEFAULT_HISTORY_TURNS))
    # Filter to valid roles and keep the last N turns.
    filtered = [
        {"role": m["role"], "content": str(m.get("content") or "")}
        for m in conversation
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and m.get("content")
    ]
    messages.extend(filtered[-history_turns:])

    return messages


def _post_tool_event(
    owner_segment: str,
    owner_id: int,
    job_id: int,
    name: str,
    args_label: str,
    status: str,
    summary: str = "",
    entity: Optional[Dict[str, Any]] = None,
    kind: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    confirm_label: Optional[str] = None,
    cancel_label: Optional[str] = None,
) -> None:
    """Best-effort POST to /assistants/:id/tool-event. Lets the UI render a
    "Searching..." card the instant the worker starts a tool — without it the
    user sees a 1-2s gap of "Thinking..." while the model thinks + tool runs.

    `entity` (e.g. {kind:'note', id:N, title:...}) is set on `done` events
    that created something deletable — the UI uses it to show a Delete button.

    `kind`/`payload`/`confirm_label`/`cancel_label` are set on
    `pending_confirmation` events so the frontend knows which confirm handler
    to invoke and what data to pass back."""
    url = f"{BACKEND_URL}/{owner_segment}/{owner_id}/tool-event"
    tool_payload: Dict[str, Any] = {"name": name, "args": args_label, "summary": summary}
    if entity:
        tool_payload["entity"] = entity
    if status == "pending_confirmation":
        tool_payload["kind"] = kind or ""
        tool_payload["payload"] = payload or {}
        if confirm_label:
            tool_payload["confirmLabel"] = confirm_label
        if cancel_label:
            tool_payload["cancelLabel"] = cancel_label
    body = {
        "jobId": job_id,
        "status": status,
        "tool": tool_payload,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2):
            pass
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning("assistant-chat: tool-event POST failed: %s", e)


# Stopwords that should never be sent to the per-token fallback search —
# they would either return everything or just add noise. Bilingual on purpose
# because the assistant accepts both languages.
_SEARCH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en",
    "y", "o", "para", "por", "con", "sin", "sobre", "que", "qué", "como",
    "cómo", "es", "ha", "han", "tengo", "tiene", "tienen", "este", "esta",
    "estos", "estas", "mi", "mis", "tu", "tus", "su", "sus", "lo", "le", "se",
    "ya", "muy", "más", "menos", "a", "al", "ni", "no", "si", "sí", "qué",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and",
    "or", "but", "is", "are", "was", "were", "be", "have", "has", "had",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "what", "which", "who", "where", "when", "why", "how",
}


def _call_search(term: str) -> List[Dict[str, Any]]:
    """One call to backend POST /search. Returns raw items list (possibly empty)."""
    if not term:
        return []
    body = json.dumps({"term": term}).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_URL}/search", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8")) or []
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("assistant-chat: search term=%r failed: %s", term, e)
        return []


def _post_backend_search(query: str) -> Dict[str, Any]:
    """Hit backend /search with progressively wider queries until we get hits
    or run out of useful terms. The model often writes natural-language queries
    like "documents about research" — the backend does literal ILIKE matching,
    so the full phrase rarely matches but individual content words do. We:

    1. Try the full phrase first (preserves precision if it actually matches).
    2. If empty, retry each non-stopword token, longest first (rarest words
       tend to be most distinctive).
    3. Merge hits dedup'd by (collection, id), trim to 10."""
    query = (query or "").strip()
    if not query:
        return {"query": query, "results": []}

    items = _call_search(query)

    if not items:
        tokens = [
            t for t in re.split(r"[\s,.;:!?¿¡()\"']+", query.lower())
            if t and t not in _SEARCH_STOPWORDS and len(t) >= 3
        ]
        # Longest first — they're typically the most specific.
        tokens.sort(key=len, reverse=True)
        seen: set = set()
        merged: List[Dict[str, Any]] = []
        for tok in tokens[:4]:
            for it in _call_search(tok):
                key = (it.get("collection"), it.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(it)
        items = merged

    # Compact shape for the model — drop scoring noise and highlight HTML.
    trimmed = []
    for it in items[:10]:
        trimmed.append({
            "collection": it.get("collection"),
            "id": it.get("id"),
            "name": it.get("name"),
        })
    return {"query": query, "results": trimmed}


# Qwen3 emits tool calls inline as <tool_call>{"name":..., "arguments":{...}}</tool_call>.
# llama-cpp-python doesn't always parse these into the OpenAI-compatible
# `tool_calls` field, so we extract them ourselves from the message content
# as a fallback. Each match yields a synthetic tool_call dict with the same
# shape the official path would have produced.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_inline_tool_calls(content: str) -> List[Dict[str, Any]]:
    if not content or "<tool_call>" not in content:
        return []
    out: List[Dict[str, Any]] = []
    for i, match in enumerate(_TOOL_CALL_RE.finditer(content)):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = str(obj.get("name") or "").strip()
        if not name:
            continue
        arguments = obj.get("arguments")
        # Tool API expects `arguments` as a JSON string, not a dict.
        args_str = (
            arguments if isinstance(arguments, str)
            else json.dumps(arguments or {}, ensure_ascii=False)
        )
        out.append({
            "id": f"inline_call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    return out


def _http_json(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Tiny JSON HTTP helper for tool dispatchers. Returns the parsed response
    or None on failure (failure is logged, never raised)."""
    url = f"{BACKEND_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("assistant-chat: %s %s failed: %s", method, path, e)
        return None


def _http_json_with_status(
    method: str, path: str, body: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Optional[Any]]:
    """Variant of _http_json that returns (status_code, parsed_body). Lets the
    caller distinguish e.g. 409 (conflict) from a hard failure."""
    url = f"{BACKEND_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8")
            return resp.getcode(), (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8")
            parsed = json.loads(raw) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        return e.code, parsed
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("assistant-chat: %s %s failed: %s", method, path, e)
        return 0, None


def _execute_create_note(args: Dict[str, Any]) -> Dict[str, Any]:
    title = str(args.get("title") or "").strip()
    body = str(args.get("body") or "").strip()
    if not title:
        return {"error": "title required"}
    payload: Dict[str, Any] = {"title": title[:200], "content": body}
    if isinstance(args.get("projectId"), int):
        payload["projectId"] = args["projectId"]
    note = _http_json("POST", "/notes", payload)
    if not isinstance(note, dict) or "id" not in note:
        return {"error": "could not create note"}
    return {
        "ok": True,
        "note": {"id": note["id"], "title": note.get("title") or title},
    }


def _execute_create_task(args: Dict[str, Any]) -> Dict[str, Any]:
    title = str(args.get("title") or "").strip()
    if not title:
        return {"error": "title required"}
    payload: Dict[str, Any] = {"title": title[:200]}
    desc = str(args.get("description") or "").strip()
    if desc:
        payload["description"] = desc
    if isinstance(args.get("projectId"), int):
        payload["projectId"] = args["projectId"]
    task = _http_json("POST", "/user-tasks", payload)
    if not isinstance(task, dict) or "id" not in task:
        return {"error": "could not create task"}
    return {
        "ok": True,
        "task": {"id": task["id"], "title": task.get("title") or title},
    }


def _normalize_alarm(value: Any) -> Optional[Dict[str, Any]]:
    """Coerce an LLM-provided alarm into the AlarmDescriptor shape, or return
    None to signal absence. Returns the sentinel string 'invalid' if it cannot
    be made valid."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return "invalid"  # type: ignore[return-value]
    offset = value.get("offsetMinutes")
    if isinstance(offset, bool) or not isinstance(offset, int):
        return "invalid"  # type: ignore[return-value]
    if offset < -10080 or offset > 0:
        return "invalid"  # type: ignore[return-value]
    out: Dict[str, Any] = {"offsetMinutes": offset}
    label = value.get("label")
    if isinstance(label, str) and label.strip():
        out["label"] = label.strip()[:100]
    return out


def _build_calendar_payload(args: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Map LLM tool args → backend create/update payload. Returns (payload, err)."""
    payload: Dict[str, Any] = {}
    if "title" in args:
        title = str(args.get("title") or "").strip()
        if not title:
            return None, "title required"
        payload["title"] = title[:200]
    if "startAt" in args:
        start = str(args.get("startAt") or "").strip()
        if not start:
            return None, "startAt required"
        payload["startDate"] = start
    if "endAt" in args and args.get("endAt") is not None:
        payload["endDate"] = str(args.get("endAt"))
    if "recurrenceRule" in args:
        rr = args.get("recurrenceRule")
        if rr is None:
            payload["recurrenceRule"] = None
        else:
            rr_s = str(rr).strip()
            if not rr_s:
                payload["recurrenceRule"] = None
            elif not rr_s.startswith("FREQ="):
                return None, "recurrenceRule must start with 'FREQ='"
            else:
                payload["recurrenceRule"] = rr_s
    if "alarm" in args:
        normalized = _normalize_alarm(args.get("alarm"))
        if normalized == "invalid":
            return None, "alarm.offsetMinutes must be integer in [-10080, 0]"
        payload["alarm"] = normalized
    if isinstance(args.get("projectId"), int):
        payload["projectId"] = args["projectId"]
    return payload, None


def _execute_create_calendar_event(args: Dict[str, Any]) -> Dict[str, Any]:
    if not str(args.get("title") or "").strip():
        return {"error": "title required"}
    if not str(args.get("startAt") or "").strip():
        return {"error": "startAt required"}
    payload, err = _build_calendar_payload(args)
    if err:
        return {"error": err}
    status, body = _http_json_with_status("POST", "/calendar-events", payload)
    if status >= 400 or not isinstance(body, dict) or "id" not in body:
        detail = body.get("message") if isinstance(body, dict) else None
        return {"error": "could not create event", "detail": detail, "status": status}
    return {
        "ok": True,
        "event": {
            "id": body["id"],
            "title": body.get("title"),
            "startDate": body.get("startDate"),
            "recurrenceRule": body.get("recurrenceRule"),
            "alarm": body.get("alarm"),
        },
    }


def _resolve_calendar_event(args: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Resolve a target event id. Returns (eventId, errorPayload). If eventId is
    set, errorPayload is None. If errorPayload is set, the caller forwards it to
    the model."""
    explicit = args.get("eventId")
    if isinstance(explicit, int):
        return explicit, None
    match = str(args.get("match") or "").strip()
    if not match:
        return None, {"error": "missing_identifier", "hint": "Provide eventId or match."}
    needle = match.lower()
    events = _http_json("GET", "/calendar-events")
    if not isinstance(events, list):
        return None, {"error": "lookup_failed"}
    candidates: List[Dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        title = str(e.get("title") or "")
        if needle in title.lower():
            candidates.append({
                "id": e.get("id"),
                "title": title,
                "startDate": e.get("startDate"),
            })
    if not candidates:
        return None, {"error": "not_found", "match": match}
    if len(candidates) > 1:
        return None, {
            "error": "ambiguous",
            "candidates": candidates[:10],
            "hint": "Ask the user which event, then retry with eventId.",
        }
    return candidates[0]["id"], None


def _execute_update_calendar_event(args: Dict[str, Any]) -> Dict[str, Any]:
    event_id, err = _resolve_calendar_event(args)
    if err is not None:
        return err
    # Strip resolution keys before mapping.
    payload_args = {k: v for k, v in args.items() if k not in ("eventId", "match")}
    if not payload_args:
        return {"error": "nothing_to_update"}
    payload, build_err = _build_calendar_payload(payload_args)
    if build_err:
        return {"error": build_err}
    status, body = _http_json_with_status("PATCH", f"/calendar-events/{event_id}", payload)
    if status >= 400 or not isinstance(body, dict) or "id" not in body:
        detail = body.get("message") if isinstance(body, dict) else None
        return {"error": "could not update event", "detail": detail, "status": status}
    return {
        "ok": True,
        "event": {
            "id": body["id"],
            "title": body.get("title"),
            "startDate": body.get("startDate"),
            "recurrenceRule": body.get("recurrenceRule"),
            "alarm": body.get("alarm"),
        },
    }


def _execute_delete_calendar_event(args: Dict[str, Any]) -> Dict[str, Any]:
    event_id, err = _resolve_calendar_event(args)
    if err is not None:
        return err
    status, body = _http_json_with_status("DELETE", f"/calendar-events/{event_id}")
    if status >= 400:
        detail = body.get("message") if isinstance(body, dict) else None
        return {"deleted": False, "error": "could not delete event", "detail": detail, "status": status}
    if isinstance(body, dict) and body.get("deleted") is False:
        return {"deleted": False, "error": "not_found", "eventId": event_id}
    return {"deleted": True, "eventId": event_id}


def _execute_get_resource_content(args: Dict[str, Any]) -> Dict[str, Any]:
    rid = args.get("resourceId")
    if not isinstance(rid, int):
        return {"error": "integer resourceId required"}
    data = _http_json("GET", f"/resources/{rid}/content")
    if not isinstance(data, dict):
        return {"error": "resource not found"}
    content = data.get("content")
    if not isinstance(content, str):
        return {"resourceId": rid, "content": None, "note": "no extracted content"}
    # Cap the content to keep the prompt manageable. The model can ask for
    # more (paged read) in a follow-up if we ever expose it.
    MAX_CONTENT_CHARS = 6000
    truncated = len(content) > MAX_CONTENT_CHARS
    return {
        "resourceId": rid,
        "content": content[:MAX_CONTENT_CHARS],
        "truncated": truncated,
    }


def _execute_list_projects(_args: Dict[str, Any]) -> Dict[str, Any]:
    data = _http_json("GET", "/projects")
    if not isinstance(data, list):
        return {"projects": []}
    projects = [
        {"id": p.get("id"), "name": p.get("name")}
        for p in data
        if isinstance(p, dict) and p.get("id")
    ]
    return {"projects": projects[:30]}


def _execute_list_notes(args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args.get("projectId")
    path = f"/notes/project/{int(pid)}" if isinstance(pid, int) else "/notes"
    data = _http_json("GET", path)
    if not isinstance(data, list):
        return {"notes": []}
    notes = []
    for n in data[:30]:
        if not isinstance(n, dict) or not n.get("id"):
            continue
        # Trim content to keep prompt size manageable. If the user asks the
        # assistant about a specific note, it can fetch its full body via a
        # follow-up tool — for now a short preview is enough to reason about.
        body = (n.get("content") or "").strip()
        preview = body[:160] + ("…" if len(body) > 160 else "")
        notes.append({
            "id": n["id"],
            "title": n.get("title") or "",
            "preview": preview,
            "projectId": (n.get("project") or {}).get("id"),
        })
    return {"notes": notes}


def _execute_list_tasks(args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args.get("projectId")
    path = f"/user-tasks/project/{int(pid)}" if isinstance(pid, int) else "/user-tasks"
    data = _http_json("GET", path)
    if not isinstance(data, list):
        return {"tasks": []}
    status_filter = str(args.get("status") or "").strip().lower() or None
    tasks = []
    for t in data:
        if not isinstance(t, dict) or not t.get("id"):
            continue
        if status_filter and t.get("status") != status_filter:
            continue
        tasks.append({
            "id": t["id"],
            "title": t.get("title") or "",
            "status": t.get("status"),
            "projectId": (t.get("project") or {}).get("id"),
        })
    return {"tasks": tasks[:50]}


def _resolve_user_task(args: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Resolve a target task id. Explicit taskId wins; otherwise titleQuery
    does a case-insensitive substring search over GET /user-tasks. Returns
    (taskId, errorPayload) — errorPayload is forwarded to the model on
    missing identifier, lookup failure, not_found, or ambiguous match."""
    explicit = args.get("taskId")
    if isinstance(explicit, int):
        return explicit, None
    query = str(args.get("titleQuery") or "").strip()
    if not query:
        return None, {"error": "missing_identifier", "hint": "Provide taskId or titleQuery."}
    needle = query.lower()
    tasks = _http_json("GET", "/user-tasks")
    if not isinstance(tasks, list):
        return None, {"error": "lookup_failed"}
    candidates: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        title = str(t.get("title") or "")
        if needle in title.lower():
            candidates.append({
                "id": t.get("id"),
                "title": title,
                "status": t.get("status"),
            })
    if not candidates:
        return None, {"error": "not_found", "titleQuery": query}
    if len(candidates) > 1:
        return None, {
            "error": "ambiguous",
            "candidates": candidates[:10],
            "hint": "Ask the user which task, then retry with taskId.",
        }
    return candidates[0]["id"], None


def _execute_update_task(args: Dict[str, Any]) -> Dict[str, Any]:
    task_id, err = _resolve_user_task(args)
    if err is not None:
        return err
    # Build the PATCH body iterating only the fields explicitly present —
    # distinguish "not sent" (skip) from "sent as null" (clear).
    payload: Dict[str, Any] = {}
    if "status" in args:
        status = args.get("status")
        if status not in ("pending", "completed"):
            return {"error": "invalid_status",
                    "hint": "status must be 'pending' or 'completed'."}
        payload["status"] = status
    if "title" in args:
        title = str(args.get("title") or "").strip()
        if not title:
            return {"error": "title_required"}
        payload["title"] = title[:200]
    if "description" in args:
        desc = args.get("description")
        payload["description"] = None if desc is None else str(desc)
    if "projectId" in args:
        pid = args.get("projectId")
        if pid is not None and not isinstance(pid, int):
            return {"error": "invalid_projectId"}
        payload["projectId"] = pid
    if not payload:
        return {"error": "nothing_to_update"}
    status_code, body = _http_json_with_status(
        "PATCH", f"/user-tasks/{task_id}", payload,
    )
    if status_code >= 400 or not isinstance(body, dict) or "id" not in body:
        detail = body.get("message") if isinstance(body, dict) else None
        return {"error": "could not update task",
                "detail": detail, "status": status_code}
    return {
        "ok": True,
        "task": {
            "id": body["id"],
            "title": body.get("title"),
            "status": body.get("status"),
        },
        "changed": list(payload.keys()),
    }


def _execute_delete_task(
    args: Dict[str, Any],
    owner_segment: str,
    owner_id: Optional[int],
    job_id: Optional[int],
) -> Dict[str, Any]:
    task_id, err = _resolve_user_task(args)
    if err is not None:
        return err
    # Always GET to obtain a human-readable title for the confirmation card —
    # the resolver may have skipped the network round-trip on explicit taskId.
    task = _http_json("GET", f"/user-tasks/{task_id}")
    if not isinstance(task, dict) or "id" not in task:
        return {"error": "not_found", "taskId": task_id}
    title = task.get("title") or f"#{task_id}"

    if isinstance(job_id, int) and isinstance(owner_id, int):
        _post_tool_event(
            owner_segment, owner_id, job_id, "delete_task", title,
            status="pending_confirmation",
            summary=f"Pending: delete {title}",
            kind="task_delete",
            payload={"taskId": task_id, "title": title},
            confirm_label="Confirm delete",
            cancel_label="Cancel",
        )

    return {
        "ok": True,
        "pendingConfirmation": True,
        "taskId": task_id,
        "title": title,
    }


def _resolve_folder_target(
    args: Dict[str, Any], owner_segment: str, owner_id: int,
) -> Dict[str, Any]:
    """Resolve a `folder_*` tool argument set to a concrete IndexedFile, using
    the same flow as folder_read (by id, then by exact filename, then by
    basename). Returns `{ok: True, indexedFileId, filename}` or an error dict.
    """
    indexed_file_id = args.get("indexedFileId")
    filename_raw = args.get("filename")

    if isinstance(indexed_file_id, int):
        path = f"/{owner_segment}/{owner_id}/indexed-files/{indexed_file_id}/content"
    elif isinstance(filename_raw, str) and filename_raw.strip():
        encoded = urllib.parse.quote(filename_raw.strip(), safe="")
        path = f"/{owner_segment}/{owner_id}/indexed-files/by-filename?filename={encoded}"
    else:
        return {"error": "bad_request", "message": "indexedFileId or filename required"}

    status, body = _http_json_with_status("GET", path)
    if status == 200 and isinstance(body, dict) and body.get("ok"):
        return {
            "ok": True,
            "indexedFileId": body.get("indexedFileId"),
            "filename": body.get("filename"),
        }
    if status == 202 and isinstance(body, dict):
        return {
            "ok": True,
            "indexedFileId": body.get("indexedFileId"),
            "filename": body.get("filename"),
        }
    if status in (404, 409, 422) and isinstance(body, dict):
        return body
    return {"error": "internal", "status": status}


def _execute_folder_delete(
    args: Dict[str, Any],
    owner_segment: str,
    owner_id: Optional[int],
    job_id: Optional[int],
) -> Dict[str, Any]:
    if not isinstance(owner_id, int):
        return {"error": "internal", "message": "missing owner context"}

    target = _resolve_folder_target(args, owner_segment, owner_id)
    if not target.get("ok"):
        return target

    indexed_file_id = target.get("indexedFileId")
    filename = target.get("filename") or ""

    if isinstance(job_id, int):
        _post_tool_event(
            owner_segment, owner_id, job_id, "folder_delete", filename,
            status="pending_confirmation",
            summary=f"Pending: delete {filename}",
            kind="folder_delete",
            payload={"indexedFileId": indexed_file_id, "filename": filename},
            confirm_label="Confirm delete",
            cancel_label="Cancel",
        )

    return {
        "ok": True,
        "pendingConfirmation": True,
        "indexedFileId": indexed_file_id,
        "filename": filename,
    }


def _execute_folder_search(
    args: Dict[str, Any], owner_segment: str, owner_id: Optional[int],
) -> Dict[str, Any]:
    if not isinstance(owner_id, int):
        return {"error": "internal", "message": "missing owner context"}

    query = str(args.get("query") or "").strip()
    if len(query) < 3:
        return {"error": "query_too_short"}

    limit = args.get("limit") if isinstance(args.get("limit"), int) else 10
    encoded = urllib.parse.quote(query, safe="")
    path = (
        f"/{owner_segment}/{owner_id}/indexed-files/search"
        f"?query={encoded}&limit={limit}"
    )
    status, body = _http_json_with_status("GET", path)

    if status == 200 and isinstance(body, dict):
        hits = body.get("hits") or []
        return {"ok": True, "hits": hits, "query": query}
    if status == 409 and isinstance(body, dict) and body.get("error") == "no_folder_configured":
        return {"error": "no_folder_configured"}
    if status == 400 and isinstance(body, dict):
        return {"error": body.get("error") or "bad_request"}
    # Backend failure or vector store unavailable — empty hits with a note so
    # the model can distinguish "nothing found" from "couldn't search".
    return {
        "ok": True,
        "hits": [],
        "query": query,
        "note": "search temporarily degraded or no matches",
    }


def _execute_folder_read(
    args: Dict[str, Any], owner_segment: str, owner_id: Optional[int],
) -> Dict[str, Any]:
    if not isinstance(owner_id, int):
        return {"error": "internal", "message": "missing owner context"}

    indexed_file_id = args.get("indexedFileId")
    filename_raw = args.get("filename")

    if isinstance(indexed_file_id, int):
        path = f"/{owner_segment}/{owner_id}/indexed-files/{indexed_file_id}/content"
    elif isinstance(filename_raw, str) and filename_raw.strip():
        # urllib.parse.quote keeps slashes by default; force quoting all chars
        # so the filename arrives intact even when it contains spaces or '#'.
        encoded = urllib.parse.quote(filename_raw.strip(), safe="")
        path = f"/{owner_segment}/{owner_id}/indexed-files/by-filename?filename={encoded}"
    else:
        return {"error": "bad_request", "message": "indexedFileId or filename required"}

    status, body = _http_json_with_status("GET", path)

    if status == 200 and isinstance(body, dict):
        return body
    if status in (404, 409, 422) and isinstance(body, dict):
        return body
    if status == 202 and isinstance(body, dict):
        return body
    return {"error": "internal", "status": status}


def _execute_folder_write(
    args: Dict[str, Any],
    owner_segment: str,
    owner_id: Optional[int],
    job_id: Optional[int],
) -> Dict[str, Any]:
    if not isinstance(owner_id, int):
        return {"error": "internal", "message": "missing owner context"}

    filename_raw = str(args.get("filename") or "")
    content = args.get("content")
    if not isinstance(content, str):
        return {"error": "internal", "message": "content must be a string"}

    # Categorize the target so we know whether the content needs conversion
    # (markdown → pdf/docx/odt, csv → xlsx) or is written verbatim.
    try:
        filename, category = normalize_and_categorize(filename_raw)
    except UnsupportedExtension as e:
        return {
            "error": "unsupported_extension",
            "filename": filename_raw,
            "extension": e.ext,
            "hint": (
                "Supported extensions: text files (.md, .txt, .csv, .json, "
                ".yaml, code…), documents (.pdf, .docx, .odt) where content "
                "must be markdown, and spreadsheets (.xlsx) where content "
                "must be CSV."
            ),
        }

    # Convert content to bytes according to category. For text we keep the
    # string and let the backend write UTF-8; for binary we send base64.
    body_payload: Dict[str, Any] = {"filename": filename}
    if category == "text":
        body_payload["content"] = content
    else:
        try:
            data = _writer_to_bytes(content, filename, category)
        except ConversionError as e:
            hint = {
                "pandoc_not_available": "Install pypandoc-binary in the worker.",
                "typst_not_available": "Install the typst pip package in the worker.",
                "openpyxl_not_available": "Install openpyxl in the worker.",
                "conversion_failed": (
                    "Check that the markdown is well-formed. For .pdf try "
                    "simpler markdown; for .xlsx ensure the content is valid "
                    "CSV with consistent column counts."
                ),
                "csv_parse_error": "The content was not valid CSV.",
            }.get(e.reason, "")
            return {
                "error": e.reason,
                "filename": filename,
                "detail": e.detail[:200] if e.detail else "",
                "hint": hint,
            }
        body_payload["contentBase64"] = base64.b64encode(data).decode("ascii")

    overwrite_requested = bool(args.get("overwrite"))

    # Always try create first. If the file does not exist, this is a clean
    # create with no destructive side-effect, regardless of `overwrite`.
    status, body = _http_json_with_status(
        "POST",
        f"/{owner_segment}/{owner_id}/indexed-files",
        body_payload,
    )

    if status == 201 and isinstance(body, dict):
        return {
            "ok": True,
            "indexedFileId": body.get("id"),
            "filename": body.get("filename") or filename,
            "category": category,
        }

    if status == 409 and isinstance(body, dict) and body.get("error") == "file_exists":
        if not overwrite_requested:
            return {"error": "file_exists", "filename": filename}
        # Overwrite intended: hand the action to the user via a confirmation
        # card. The frontend's folder_overwrite handler reposts with overwrite.
        # For binary categories we ship contentBase64 in the payload so the
        # frontend doesn't have to re-run pandoc/openpyxl.
        if isinstance(job_id, int):
            payload: Dict[str, Any] = {"filename": filename}
            if "content" in body_payload:
                payload["content"] = body_payload["content"]
            else:
                payload["contentBase64"] = body_payload["contentBase64"]
            _post_tool_event(
                owner_segment, owner_id, job_id, "folder_write", filename,
                status="pending_confirmation",
                summary=f"Pending: overwrite {filename}",
                kind="folder_overwrite",
                payload=payload,
                confirm_label="Confirm overwrite",
                cancel_label="Cancel",
            )
        return {"ok": True, "pendingConfirmation": True, "filename": filename}

    if status == 409 and isinstance(body, dict):
        return {"error": body.get("error") or "conflict", "filename": filename}
    if status == 400 and isinstance(body, dict):
        err = body.get("error") or body.get("message") or "bad_request"
        return {"error": err, "filename": filename}
    if status == 403 and isinstance(body, dict):
        return {"error": body.get("error") or "forbidden", "filename": filename}
    return {"error": "internal", "filename": filename, "status": status}


def _execute_tool(
    name: str,
    args_json: str,
    kind: str = "assistant",
    owner_segment: str = "assistants",
    owner_id: Optional[int] = None,
    job_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Dispatch a single tool call. Returns the result as a dict (will be
    JSON-encoded by the caller before feeding back to the model).

    Defensive: for agents, reject any tool name outside AGENT_ALLOWED_TOOLS
    even though they are pre-filtered out of the tools list seen by the model.
    Covers hallucinations where the model invents a tool name."""
    if kind == "agent" and name not in AGENT_ALLOWED_TOOLS:
        logger.warning(
            "assistant-chat: rejected tool '%s' for agent (not in allowlist)", name,
        )
        return {"error": "tool_not_allowed_for_agent", "tool": name}

    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    if name == "search_workspace":
        return _post_backend_search(str(args.get("query") or "").strip())
    if name == "create_note":
        return _execute_create_note(args)
    if name == "create_task":
        return _execute_create_task(args)
    if name == "create_calendar_event":
        return _execute_create_calendar_event(args)
    if name == "update_calendar_event":
        return _execute_update_calendar_event(args)
    if name == "delete_calendar_event":
        return _execute_delete_calendar_event(args)
    if name == "get_resource_content":
        return _execute_get_resource_content(args)
    if name == "list_projects":
        return _execute_list_projects(args)
    if name == "list_notes":
        return _execute_list_notes(args)
    if name == "list_tasks":
        return _execute_list_tasks(args)
    if name == "update_task":
        return _execute_update_task(args)
    if name == "delete_task":
        return _execute_delete_task(args, owner_segment, owner_id, job_id)
    if name == "folder_search":
        return _execute_folder_search(args, owner_segment, owner_id)
    if name == "folder_read":
        return _execute_folder_read(args, owner_segment, owner_id)
    if name == "folder_write":
        return _execute_folder_write(args, owner_segment, owner_id, job_id)
    if name == "folder_delete":
        return _execute_folder_delete(args, owner_segment, owner_id, job_id)
    return {"error": f"Unknown tool: {name}"}


def _run_tool_rounds(
    llm,
    messages: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    payload: Dict[str, Any],
    job_id: Optional[int] = None,
) -> None:
    """Iterate tool-call rounds until the model produces a plain text reply or
    we hit MAX_TOOL_ROUNDS. Tool events are emitted LIVE via POST /tool-event
    (so the UI shows 'Searching...' the instant the tool starts) — this
    function no longer returns events to the caller."""
    tool_max_tokens = int(cfg.get("tool_max_tokens", 600))
    kind = (payload.get("kind") or "assistant")
    owner_id = _owner_id(payload)
    owner_segment = _backend_owner_segment(payload)
    can_emit_live = isinstance(owner_id, int) and isinstance(job_id, int)
    tools_for_model = _tools_for_payload(payload)
    tool_names = [t.get("function", {}).get("name") for t in tools_for_model]
    logger.info(
        "assistant-chat: tool round input kind=%s owner=%s/%s tools=%s",
        kind, owner_segment, owner_id, tool_names,
    )
    logger.info(
        "assistant-chat: tool round input — last_user=%r system_head=%r",
        next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")[:120],
        next((m.get("content") for m in messages if m.get("role") == "system"), "")[:160],
    )
    for round_idx in range(MAX_TOOL_ROUNDS):
        msg = llm.chat_with_tools(messages, tools_for_model, max_tokens=tool_max_tokens)
        content = _strip_thinking(msg.get("content") or "")
        tool_calls = msg.get("tool_calls") or []
        # Qwen3 often emits the call inline (<tool_call>…</tool_call>) instead
        # of populating tool_calls. Fall back to extracting from content.
        if not tool_calls and content:
            inline = _extract_inline_tool_calls(content)
            if inline:
                tool_calls = inline
        logger.info(
            "assistant-chat: tool round %d → tool_calls=%d content_head=%r",
            round_idx, len(tool_calls), content[:120],
        )
        if not tool_calls:
            # Model is ready to give the user a direct reply. We DON'T keep
            # this content — the final streaming call regenerates it.
            return
        # Persist the assistant message that triggered the calls — the
        # chat template needs it for the next turn to make sense.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or None,
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args_json = fn.get("arguments") or "{}"
            # Pull a short human-readable label from the args (e.g. the query).
            try:
                args_obj = json.loads(args_json or "{}")
                args_label = str(args_obj.get("query") or next(iter(args_obj.values()), ""))
            except (json.JSONDecodeError, StopIteration):
                args_label = ""
            # LIVE: emit the card the instant we know which tool the model
            # picked, before running it. The UI flips from "Thinking..." to
            # "Searching '<query>'..." immediately.
            if can_emit_live:
                _post_tool_event(
                    owner_segment, owner_id, job_id, name, args_label,
                    status="running",
                )
            result = _execute_tool(
                name, args_json, kind=kind,
                owner_segment=owner_segment, owner_id=owner_id, job_id=job_id,
            )
            # Per-tool result summary + extract any created entity so the
            # frontend can render a "Delete" action on the card.
            entity: Optional[Dict[str, Any]] = None
            if isinstance(result, dict):
                if name == "search_workspace":
                    hits = len(result.get("results") or [])
                    summary = f"{hits} results"
                elif name == "list_projects":
                    n = len(result.get("projects") or [])
                    summary = f"{n} projects"
                elif name == "list_notes":
                    n = len(result.get("notes") or [])
                    summary = f"{n} notes"
                elif name == "list_tasks":
                    n = len(result.get("tasks") or [])
                    summary = f"{n} tasks"
                elif name == "create_note" and isinstance(result.get("note"), dict):
                    note = result["note"]
                    summary = f"Note: {note.get('title') or ''}"
                    entity = {"kind": "note", "id": note.get("id"), "title": note.get("title")}
                elif name == "create_task" and isinstance(result.get("task"), dict):
                    task = result["task"]
                    summary = f"Task: {task.get('title') or ''}"
                    entity = {"kind": "task", "id": task.get("id"), "title": task.get("title")}
                elif name == "update_task" and isinstance(result.get("task"), dict):
                    task = result["task"]
                    changed = result.get("changed") or []
                    if "status" in changed:
                        verb = "Done" if task.get("status") == "completed" else "Re-opened"
                        summary = f"{verb}: {task.get('title') or ''}"
                    elif "title" in changed:
                        summary = f"Renamed: {task.get('title') or ''}"
                    elif "description" in changed:
                        summary = f"Description updated: {task.get('title') or ''}"
                    elif "projectId" in changed:
                        summary = f"Project changed: {task.get('title') or ''}"
                    else:
                        summary = f"Updated: {task.get('title') or ''}"
                    entity = {"kind": "task", "id": task.get("id"), "title": task.get("title")}
                elif name == "create_calendar_event" and isinstance(result.get("event"), dict):
                    ev = result["event"]
                    summary = f"Event: {ev.get('title') or ''}"
                    entity = {"kind": "calendarEvent", "id": ev.get("id"), "title": ev.get("title")}
                elif name == "update_calendar_event" and isinstance(result.get("event"), dict):
                    ev = result["event"]
                    summary = f"Updated: {ev.get('title') or ''}"
                    entity = {"kind": "calendarEvent", "id": ev.get("id"), "title": ev.get("title")}
                elif name == "delete_calendar_event" and result.get("deleted") is True:
                    summary = f"Deleted event #{result.get('eventId')}"
                elif name == "get_resource_content":
                    n = len(result.get("content") or "")
                    summary = f"{n} chars read" + (" (truncated)" if result.get("truncated") else "")
                elif name == "folder_write" and result.get("ok"):
                    fn = result.get("filename") or ""
                    summary = f"Wrote: {fn}"
                    entity = {
                        "kind": "indexedFile",
                        "id": result.get("indexedFileId"),
                        "title": fn,
                    }
                elif name == "folder_read" and result.get("ok"):
                    fn = result.get("filename") or ""
                    summary = f"Read: {fn}"
                    if result.get("derivedFromExtraction"):
                        summary += " (extracted)"
                elif name == "folder_search" and result.get("ok"):
                    hits = result.get("hits") or []
                    q = result.get("query") or ""
                    if hits:
                        summary = f"{len(hits)} file(s) for «{q}»"
                    else:
                        summary = f"No matches for «{q}»"
                elif "error" in result:
                    summary = f"error: {result['error']}"
                else:
                    summary = "OK"
            else:
                summary = "OK"
            logger.info("assistant-chat: tool=%s args=%s → %s",
                        name, args_json[:120], summary)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
            # LIVE: finalise the card with the result count + any created entity.
            # Tools that emit their own pending_confirmation card own the final
            # state — don't overwrite it with a generic `done` here.
            emitted_pending = (
                isinstance(result, dict) and result.get("pendingConfirmation")
            )
            if can_emit_live and not emitted_pending:
                _post_tool_event(
                    owner_segment, owner_id, job_id, name, args_label,
                    status="done", summary=summary, entity=entity,
                )
    # Ran out of rounds. The streaming call will at least emit *something*
    # with the tool results in context.


@job_handler("assistant-chat")
def assistant_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cfg = get_task_config("assistant-chat")
        messages = _build_messages(payload, cfg)
        if not messages or messages[-1]["role"] != "user":
            return {"error": "History does not end with a user message"}

        max_tokens = int(cfg.get("max_tokens", 1000))
        params = get_llm_params("assistant-chat")
        llm = get_llm_service(**params)

        kind = (payload.get("kind") or "assistant")
        owner_id = _owner_id(payload)
        owner_segment = _backend_owner_segment(payload)

        logger.info(
            "assistant-chat: kind=%s owner=%s/%s name=%s turns=%d max_tokens=%d",
            kind, owner_segment, owner_id,
            payload.get("name") or payload.get("assistantName"),
            len(messages),
            max_tokens,
        )

        # IDs are needed for both tool live events and streaming chunks.
        job_id = payload.get("jobId")

        # Tool-call phase. The personal assistant gets the full ASSISTANT_TOOLS
        # list; agents get only the folder_* subset (filtered inside
        # _run_tool_rounds via _tools_for_payload). The phase is non-streaming
        # because the model has to decide whether to call a tool BEFORE it
        # produces user-visible text. If a tool runs, its result gets appended
        # to `messages` and the streaming phase below sees the augmented
        # conversation as context. Tool cards are pushed LIVE to the UI via
        # POST /tool-event from inside _run_tool_rounds.
        #
        # Tool rounds are entered for: the personal assistant (assistantSystem)
        # and any agent (always — they only see folder_*, but that's exactly
        # what they need to do their job).
        if payload.get("assistantSystem") or kind == "agent":
            logger.info("assistant-chat: entering tool phase (kind=%s)", kind)
            _run_tool_rounds(llm, messages, cfg, payload, job_id)
        else:
            logger.info(
                "assistant-chat: skipping tool phase (assistantSystem=%r kind=%s)",
                payload.get("assistantSystem"), kind,
            )
        can_stream = (
            bool(cfg.get("stream", True))
            and isinstance(owner_id, int)
            and isinstance(job_id, int)
        )

        raw_parts: List[str] = []
        if can_stream:
            logger.info(
                "assistant-chat: streaming enabled (kind=%s owner=%s/%s job=%s)",
                kind, owner_segment, owner_id, job_id,
            )
            # Streaming state machine for <think>...</think> blocks: even with
            # `/no_think`, Qwen3 sometimes opens an empty pair at the start.
            # We accumulate everything in raw_parts (kept verbatim for the
            # final reply, where _strip_thinking handles it), but ONLY forward
            # chunks to the UI when we're outside any thinking block.
            visible_pending = ""        # characters seen but not yet decided
            in_think = False
            buffer: List[str] = []
            chunks_sent = 0
            last_flush_ms = time.monotonic() * 1000

            def _process_for_ui(piece: str) -> str:
                """Return the user-visible portion of this chunk, given current
                thinking state. Updates in_think / visible_pending across calls."""
                nonlocal in_think, visible_pending
                out_parts: List[str] = []
                buf = visible_pending + piece
                while buf:
                    if in_think:
                        end = buf.find("</think>")
                        if end == -1:
                            # Still inside thinking — swallow everything, keep
                            # nothing in pending (we never emit thinking text).
                            buf = ""
                            break
                        buf = buf[end + len("</think>"):]
                        in_think = False
                    else:
                        start = buf.find("<think>")
                        if start == -1:
                            # No new opening tag. But the tail might be a
                            # partial "<think" — hold it back to avoid leaking.
                            for i in range(1, min(len("<think>"), len(buf)) + 1):
                                if "<think>".startswith(buf[-i:]):
                                    out_parts.append(buf[:-i])
                                    visible_pending = buf[-i:]
                                    return "".join(out_parts)
                            out_parts.append(buf)
                            visible_pending = ""
                            return "".join(out_parts)
                        # Emit text before <think>, swallow the tag, enter thinking.
                        out_parts.append(buf[:start])
                        buf = buf[start + len("<think>"):]
                        in_think = True
                visible_pending = ""
                return "".join(out_parts)

            for piece in llm.chat_stream(messages, max_tokens=max_tokens):
                raw_parts.append(piece)
                visible = _process_for_ui(piece)
                if visible:
                    buffer.append(visible)
                now_ms = time.monotonic() * 1000
                if buffer and now_ms - last_flush_ms >= STREAM_FLUSH_INTERVAL_MS:
                    _post_stream_chunk(owner_segment, owner_id, job_id, "".join(buffer))
                    chunks_sent += 1
                    buffer.clear()
                    last_flush_ms = now_ms
            if buffer:
                _post_stream_chunk(owner_segment, owner_id, job_id, "".join(buffer))
                chunks_sent += 1
            # Final marker so the UI can stop the caret immediately, even
            # though memory extraction below may still take a beat.
            _post_stream_chunk(owner_segment, owner_id, job_id, "", done=True)
            raw = "".join(raw_parts)
            logger.info("assistant-chat: stream done, %d chunks sent", chunks_sent)
        else:
            logger.warning(
                "assistant-chat: streaming disabled (kind=%s owner=%s/%s job=%r stream_cfg=%r)",
                kind, owner_segment, owner_id, job_id, cfg.get("stream"),
            )
            raw = llm.chat(messages, max_tokens=max_tokens) or ""

        reply = _strip_thinking(raw)
        if not reply:
            return {"error": "Model returned an empty response"}

        result: Dict[str, Any] = {"reply": reply}

        # Tool events are no longer returned here — they're emitted live by
        # _run_tool_rounds via POST /tool-event so the UI sees them in real
        # time, not at job completion.

        # Memory management is only relevant for the personal assistant.
        # Agents have no memory; skip the entire block.
        if kind == "assistant" and payload.get("extractMemory"):
            user_message = _last_user_message(payload)
            if user_message:
                action = _extract_memory_action(
                    llm,
                    user_message,
                    payload.get("memorySnippets") or [],
                    cfg,
                )
                if action:
                    result["memoryAction"] = action
                    logger.info("assistant-chat: memoryAction=%r", action)

        return result
    except Exception as e:  # noqa: BLE001
        logger.exception("assistant-chat handler failed")
        return {"error": f"Assistant failure: {e}"}
