import argparse
import json
import os
import statistics
import time
import urllib.request
import uuid
from pathlib import Path
from unittest.mock import patch

import psycopg
from psycopg.rows import dict_row

from agents.loop import run_agent_loop
from config import (
    EXECUTIONS_TABLE,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from lib.execution import ExecutionEmitter, activate_emitter, reset_emitter
from lib.framework.agent import AgentSpec
from lib.framework.tool import ToolContext
from lib.llm.config import get_llm_params
from services.llm_service import get_llm_service
from tests.execution.benchmark_progress_overhead import (
    SCENARIOS as MVP04_SCENARIOS,
    TOOLS,
    CountingLlm,
    dispatch_for,
)


SCENARIO_NAMES = (
    "direct_with_catalog",
    "one_tool",
    "output_repair",
    "forced_finalization",
)

EXPECTED_BUDGET_USAGE = {
    "direct_with_catalog": {"normal": 1, "repair": 0, "closing": 0},
    "one_tool": {"normal": 2, "repair": 0, "closing": 0},
    "output_repair": {"normal": 1, "repair": 1, "closing": 0},
    "forced_finalization": {"normal": 1, "repair": 0, "closing": 1},
}


class Mvp04Emitter(ExecutionEmitter):
    def request_progress_grant(self, _request):
        return {}


def request(backend_url, method, path, body=None):
    value = urllib.request.Request(
        f"{backend_url}{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Workspace-Id": "mvp05-benchmark",
        },
    )
    with urllib.request.urlopen(value, timeout=10) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def database_connection():
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


def create_emitter(backend_url, assistant_id, budget_protocol):
    if budget_protocol:
        created = request(
            backend_url,
            "POST",
            f"/assistants/{assistant_id}/messages",
            {"content": "MVP 05 inference budget benchmark probe"},
        )
        emitter_type = ExecutionEmitter
    else:
        created = request(
            backend_url,
            "POST",
            "/executions",
            {"taskType": "search", "content": "MVP 04 benchmark control"},
        )
        emitter_type = Mvp04Emitter
    context = activate_execution(created["executionId"])
    return emitter_type(context), context


def budget_usage(progress):
    grants = ((progress.get("ledger") or {}).get("inferenceBudget") or {}).get(
        "grants", {}
    )
    if len(grants) != 1:
        return None
    grant = next(iter(grants.values()))
    return {
        bucket: int((grant["usage"][bucket] or {}).get("consumed", 0))
        for bucket in ("normal", "repair", "closing")
    }


def run_sample(service, backend_url, assistant_id, name, budget_protocol):
    scenario = MVP04_SCENARIOS[name]
    emitter, context = create_emitter(backend_url, assistant_id, budget_protocol)
    token = activate_emitter(emitter)
    llm = CountingLlm(service, bool(scenario.get("inject_empty")))
    calls = []
    spec = AgentSpec(
        name="mvp05-benchmark",
        config_key="mvp05-benchmark",
        system_prompt=scenario["system"],
        tool_names=frozenset(scenario["tools"]),
        max_rounds=scenario["max_rounds"],
        fallback_max_tokens=96,
    )
    ctx = ToolContext(execution=emitter)
    messages = [
        {"role": "system", "content": scenario["system"]},
        {"role": "user", "content": scenario["user"]},
    ]
    started = time.perf_counter()
    try:
        with patch(
            "agents.loop.get_task_config",
            return_value={
                "max_rounds": scenario["max_rounds"],
                "max_tokens": 96,
            },
        ), patch("agents.loop.get_llm_params", return_value={}), patch(
            "agents.loop.get_llm_service", return_value=llm
        ):
            result = run_agent_loop(
                spec,
                messages,
                ctx,
                [TOOLS[item] for item in scenario["tools"]],
                lambda tool, arguments, tool_ctx: dispatch_for(
                    calls, tool, arguments, tool_ctx
                ),
            )
        if result.kind == "final_text":
            emitter.record_final_message(result.content or "")
            emitter.flush_evidence()
    finally:
        reset_emitter(token)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    content = result.content or ""
    content_matches = scenario["marker"] in content
    classification_matches = (
        result.completion_kind == "partial"
        and result.completion_reason == "budget_exhausted"
        if name == "forced_finalization"
        else result.completion_kind is None and result.completion_reason is None
    )
    if name == "forced_finalization":
        content_matches = bool(content.strip())
    correct = (
        result.kind == "final_text"
        and content_matches
        and classification_matches
        and calls == scenario["expected_tools"]
    )
    if name == "output_repair":
        correct = correct and llm.inferences >= 2
    if name == "forced_finalization":
        correct = correct and llm.inferences == 2
    usage = None
    if budget_protocol:
        progress = request(
            backend_url,
            "GET",
            f"/executions/{context['rootExecutionId']}/progress",
        )
        usage = budget_usage(progress)
        correct = correct and usage == EXPECTED_BUDGET_USAGE[name]
    return {
        "mode": "mvp05" if budget_protocol else "mvp04_control",
        "elapsedMs": elapsed_ms,
        "instrumentationMs": emitter.instrumentation_ms,
        "inferences": llm.inferences,
        "tools": calls,
        "resultKind": result.kind,
        "completionKind": result.completion_kind,
        "completionReason": result.completion_reason,
        "content": content,
        "budgetUsage": usage,
        "correct": correct,
    }


def collect_samples(service, backend_url, assistant_id, name, sample_count, max_attempts):
    attempts = []
    accepted = {"mvp04_control": 0, "mvp05": 0}
    for attempt in range(max_attempts):
        order = (False, True) if attempt % 2 == 0 else (True, False)
        for budget_protocol in order:
            mode = "mvp05" if budget_protocol else "mvp04_control"
            if accepted[mode] >= sample_count:
                continue
            sample = run_sample(
                service,
                backend_url,
                assistant_id,
                name,
                budget_protocol,
            )
            attempts.append(sample)
            print(name, json.dumps(sample, ensure_ascii=False), flush=True)
            if sample["correct"]:
                accepted[mode] += 1
        if all(value >= sample_count for value in accepted.values()):
            break
    if any(value < sample_count for value in accepted.values()):
        raise RuntimeError(f"{name}: insufficient correct samples: {accepted}")
    return attempts


def accepted(attempts, mode, key, sample_count):
    return [
        item[key]
        for item in attempts
        if item["correct"] and item["mode"] == mode
    ][:sample_count]


def summarize(attempts, name, sample_count):
    baseline = accepted(attempts, "mvp04_control", "elapsedMs", sample_count)
    governed = accepted(attempts, "mvp05", "elapsedMs", sample_count)
    baseline_median = statistics.median(baseline)
    governed_median = statistics.median(governed)
    delta = governed_median - baseline_median
    threshold = max(round(baseline_median * 0.10), 150)
    baseline_content = accepted(
        attempts, "mvp04_control", "content", sample_count
    )
    governed_content = accepted(attempts, "mvp05", "content", sample_count)
    semantic_match = set(baseline_content) == set(governed_content)
    trajectory_match = (
        accepted(attempts, "mvp04_control", "inferences", sample_count)
        == accepted(attempts, "mvp05", "inferences", sample_count)
        and accepted(attempts, "mvp04_control", "tools", sample_count)
        == accepted(attempts, "mvp05", "tools", sample_count)
    )
    return {
        "mvp04ControlMs": baseline,
        "mvp05Ms": governed,
        "mvp04ControlMedianMs": baseline_median,
        "mvp05MedianMs": governed_median,
        "deltaMs": delta,
        "thresholdMs": threshold,
        "semanticMatch": semantic_match,
        "trajectoryMatch": trajectory_match,
        "mvp05ResultKinds": accepted(
            attempts, "mvp05", "resultKind", sample_count
        ),
        "mvp05CompletionKinds": accepted(
            attempts, "mvp05", "completionKind", sample_count
        ),
        "mvp05CompletionReasons": accepted(
            attempts, "mvp05", "completionReason", sample_count
        ),
        "mvp05BudgetUsage": accepted(
            attempts, "mvp05", "budgetUsage", sample_count
        ),
        "expectedBudgetUsage": EXPECTED_BUDGET_USAGE[name],
        "passed": delta <= threshold and semantic_match and trajectory_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:3000").rstrip("/")
    assistant = request(
        backend_url,
        "POST",
        "/assistants",
        {"name": "MVP 05 validation", "systemPrompt": "Validation fixture"},
    )
    params = get_llm_params("assistant-chat")
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

    raw_results = {
        name: collect_samples(
            service,
            backend_url,
            assistant["id"],
            name,
            args.samples,
            args.max_attempts,
        )
        for name in SCENARIO_NAMES
    }
    scenarios = {
        name: summarize(attempts, name, args.samples)
        for name, attempts in raw_results.items()
    }
    report = {
        "model": Path(params["model_path"]).name,
        "samplesPerMode": args.samples,
        "scenarios": scenarios,
        "passed": all(value["passed"] for value in scenarios.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
