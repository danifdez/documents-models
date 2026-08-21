"""Shared plumbing for the execution benchmarks in tests/execution/.

Each benchmark owns its scenarios, its instrumented LLM and its `run_sample`;
everything here is infrastructure they must share so it cannot drift apart.
"""

import json
import os
import statistics
import urllib.request
import uuid
from pathlib import Path

from config import (
    EXECUTIONS_TABLE,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from lib.llm.config import get_llm_params
from services.llm_service import get_llm_service

DEFAULT_BACKEND_URL = "http://127.0.0.1:3000"


def ensure_ingest_token():
    if os.environ.get("EXECUTION_INGEST_TOKEN"):
        return
    env_path = Path(__file__).resolve().parents[3] / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("EXECUTION_INGEST_TOKEN="):
            os.environ["EXECUTION_INGEST_TOKEN"] = line.split("=", 1)[1]
            return
    raise RuntimeError("EXECUTION_INGEST_TOKEN is not configured for the profile")


def resolve_backend_url():
    return os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


def backend_client(workspace, timeout=30):
    """Build the benchmark's `request`, bound to its workspace and timeout.

    The workspace groups a benchmark's executions on the backend side, so it is
    a per-file identity rather than a transport detail.
    """
    def request(backend_url, method, path, body=None):
        value = urllib.request.Request(
            f"{backend_url}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Workspace-Id": workspace,
            },
        )
        with urllib.request.urlopen(value, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    return request


def create_assistant(request, backend_url, name):
    return request(
        backend_url,
        "POST",
        "/assistants",
        {"name": name, "systemPrompt": "Validation fixture"},
    )


def database_connection():
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        row_factory=dict_row,
        autocommit=True,
    )


def activate_execution(execution_id):
    attempt_id = str(uuid.uuid4())
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE {EXECUTIONS_TABLE}
            SET status = 'running', phase = 'worker_execution',
                attempt_id = %s, started_at = COALESCE(started_at, now()),
                updated_at = now()
            WHERE execution_id = %s
            RETURNING root_execution_id, execution_id, turn_id, last_event_id
            """,
            (attempt_id, execution_id),
        )
        execution = cursor.fetchone()
    if not execution:
        raise RuntimeError(f"Execution {execution_id} was not created")
    return {
        "rootExecutionId": str(execution["root_execution_id"]),
        "executionId": str(execution["execution_id"]),
        "turnId": str(execution["turn_id"]) if execution["turn_id"] else None,
        "attemptId": attempt_id,
        "causedByEventId": str(execution["last_event_id"]),
    }


def materialized_prompts(root_execution_id):
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT body
            FROM execution_artifacts
            WHERE root_execution_id = %s AND kind = 'materialized_prompt'
            ORDER BY created_at, artifact_id
            """,
            (root_execution_id,),
        )
        rows = cursor.fetchall()
    return [json.loads(bytes(row["body"]).decode("utf-8")) for row in rows]


def index_tool_schema(name, description):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "minimum": 1},
                },
                "required": ["index"],
            },
        },
    }


def deterministic_service(config_key="assistant-chat"):
    """An LLM service with sampling pinned to greedy decoding, already warm.

    Without the warm-up call the first scenario would absorb the server start-up
    cost and skew the latency comparison it is supposed to measure.
    """
    params = get_llm_params(config_key)
    service = get_llm_service(**params)
    service.sampling = {
        **service.sampling,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }
    service.chat(
        [{"role": "user", "content": "/no_think\nReply OK."}],
        max_tokens=8,
        allow_thinking=False,
    )
    return service, params


def collect_paired(sampler, name, samples, max_attempts, mode_names):
    """Collect correct samples for both modes, alternating which one runs first.

    Without the alternation the mode that always runs second benefits from the
    other's warm caches and the latency comparison is biased.
    """
    values = []
    accepted_counts = {mode: 0 for mode in mode_names}
    for attempt in range(max_attempts):
        order = (False, True) if attempt % 2 == 0 else (True, False)
        for enabled in order:
            mode = mode_names[1] if enabled else mode_names[0]
            if accepted_counts[mode] >= samples:
                continue
            value = sampler(enabled)
            values.append(value)
            print(name, json.dumps(value, ensure_ascii=False), flush=True)
            if value["correct"]:
                accepted_counts[mode] += 1
        if all(count >= samples for count in accepted_counts.values()):
            break
    if any(count < samples for count in accepted_counts.values()):
        raise RuntimeError(
            f"{name}: insufficient correct samples: {accepted_counts}"
        )
    return values


def accepted(values, mode, key, samples):
    return [
        value[key]
        for value in values
        if value["correct"] and value["mode"] == mode
    ][:samples]


def latency_block(
    values,
    samples,
    *,
    control_mode,
    current_mode,
    control_key,
    current_key,
):
    """The latency keys every `summarize` shares, in their published order."""
    control = accepted(values, control_mode, "elapsedMs", samples)
    current = accepted(values, current_mode, "elapsedMs", samples)
    control_median = statistics.median(control)
    current_median = statistics.median(current)
    return {
        f"{control_key}Ms": control,
        f"{current_key}Ms": current,
        f"{control_key}MedianMs": control_median,
        f"{current_key}MedianMs": current_median,
        "deltaMs": current_median - control_median,
        "thresholdMs": max(round(control_median * 0.10), 150),
    }


def write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)
