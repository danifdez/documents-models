"""Backend transport shared by tool executors and the parent chat loop.

Local-only by design: the worker talks to the NestJS backend over plain HTTP on
localhost to run tools and to push tool-event cards to the UI.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Backend HTTP endpoint used both for streaming chunks back to the UI and for
# executing assistant tools (search, etc.). Local-only by design.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")


def http_json(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Optional[Any]:
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


def http_json_with_status(
    method: str, path: str, body: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Optional[Any]]:
    """Variant of http_json that returns (status_code, parsed_body). Lets the
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


def post_stream_chunk(
    owner_segment: str,
    owner_id: int,
    job_id: int,
    chunk: str,
    done: bool = False,
) -> None:
    """Best-effort POST of a partial reply chunk to /:segment/:id/stream-chunk.
    Failures are logged but never raised — streaming is purely a UX nicety; the
    final reply still arrives through the normal job-result path.

    When `done=True`, signals that the model has finished generating even if the
    job hasn't fully completed (memory extraction may still run). The UI uses it
    to stop the live caret immediately."""
    if not chunk and not done:
        return
    url = f"{BACKEND_URL}/{owner_segment}/{owner_id}/stream-chunk"
    body: Dict[str, Any] = {"jobId": job_id, "chunk": chunk}
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
        logger.warning("chat: stream-chunk POST failed: %s", e)


def post_tool_event(
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
