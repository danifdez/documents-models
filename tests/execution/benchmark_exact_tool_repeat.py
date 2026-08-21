import argparse
import copy
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

from agents.loop import run_agent_loop
from config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from lib.execution import ExecutionEmitter, activate_emitter, reset_emitter
from lib.framework.agent import AgentSpec
from lib.framework.tool import Tool, ToolContext
from lib.llm.config import get_llm_params
from services.llm_service import get_llm_service
from tests.execution.benchmark_inference_budget import activate_execution
from tools import REGISTRY


REPEAT_WARNING = "An exact tool call was repeated without an intervening tool operation."


def ensure_ingest_token():
    if os.environ.get("EXECUTION_INGEST_TOKEN"):
        return
    env_path = Path(__file__).resolve().parents[3] / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("EXECUTION_INGEST_TOKEN="):
            os.environ["EXECUTION_INGEST_TOKEN"] = line.split("=", 1)[1]
            return
    raise RuntimeError("EXECUTION_INGEST_TOKEN is not configured for the profile")


def tool_schema(name, description):
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


COLLECT = tool_schema(
    "collect_evidence",
    "Collect the validation evidence for one numeric index.",
)
INSPECT = tool_schema(
    "inspect_evidence",
    "Inspect the validation evidence for one numeric index.",
)
FAIL = tool_schema(
    "failing_probe",
    "Run the validation probe. It returns an expected error.",
)


def scenario(
    marker,
    instruction,
    *,
    tools,
    max_rounds,
    expected_tools,
    injected=None,
    expected_guard=False,
    expected_warning=False,
    expected_partial=False,
    expected_post_warning=None,
    parity=True,
    max_tool_calls=6,
):
    return {
        "marker": marker,
        "system": f"/no_think\n{instruction}",
        "user": "Complete the validation sequence exactly as instructed.",
        "tools": tools,
        "maxRounds": max_rounds,
        "expectedTools": expected_tools,
        "injected": injected or [],
        "expectedGuard": expected_guard,
        "expectedWarning": expected_warning,
        "expectedPartial": expected_partial,
        "expectedPostWarning": expected_post_warning,
        "parity": parity,
        "maxToolCalls": max_tool_calls,
    }


SCENARIOS = {
    "direct": scenario(
        "MVP10-DIRECT-11",
        "Answer exactly MVP10-DIRECT-11 without using a tool.",
        tools=[],
        max_rounds=1,
        expected_tools=[],
    ),
    "one_tool": scenario(
        "MVP10-ONE-23",
        "Call collect_evidence once with index 1. After its result answer "
        "exactly MVP10-ONE-23 without another tool.",
        tools=[COLLECT],
        max_rounds=2,
        expected_tools=[["collect_evidence", 1]],
    ),
    "two_different_tools": scenario(
        "MVP10-TWO-31",
        "Call collect_evidence with index 1 and inspect_evidence with index 2 "
        "together in the first response. Then answer exactly MVP10-TWO-31.",
        tools=[COLLECT, INSPECT],
        max_rounds=2,
        expected_tools=[["collect_evidence", 1], ["inspect_evidence", 2]],
    ),
    "repeat_then_answer": scenario(
        "MVP10-RECOVER-43",
        "A test injector will first call collect_evidence twice with index 1. "
        "After those results, if a system message says an exact tool call was "
        "repeated, answer exactly MVP10-RECOVER-43. Without that warning call "
        "collect_evidence once more with index 1, then answer the marker.",
        tools=[COLLECT],
        max_rounds=4,
        injected=[("collect_evidence", 1), ("collect_evidence", 1)],
        expected_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        expected_guard=True,
        expected_warning=True,
        expected_post_warning="answer",
        parity=False,
    ),
    "repeat_then_change_arguments": scenario(
        "MVP10-CHANGE-53",
        "A test injector will first call collect_evidence twice with index 1. "
        "After those results, if a system message says an exact tool call was "
        "repeated, call collect_evidence once with index 2 and then answer "
        "exactly MVP10-CHANGE-53. Without that warning repeat index 1 once, "
        "then answer the marker.",
        tools=[COLLECT],
        max_rounds=4,
        injected=[("collect_evidence", 1), ("collect_evidence", 1)],
        expected_tools=[
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["collect_evidence", 2],
        ],
        expected_guard=True,
        expected_warning=True,
        expected_partial=True,
        expected_post_warning="changed_arguments",
        parity=False,
    ),
    "repeat_persists": scenario(
        "MVP10-PERSIST-61",
        "A test injector will first call collect_evidence twice with index 1. "
        "After those results call collect_evidence once more with index 1 even "
        "if a system warning asks you to change strategy. Then answer exactly "
        "MVP10-PERSIST-61.",
        tools=[COLLECT],
        max_rounds=4,
        injected=[("collect_evidence", 1), ("collect_evidence", 1)],
        expected_tools=[
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["collect_evidence", 1],
        ],
        expected_guard=True,
        expected_warning=True,
        expected_partial=True,
        expected_post_warning="repeated",
        parity=False,
    ),
    "different_arguments_from_start": scenario(
        "MVP10-DIFFERENT-67",
        "Call collect_evidence first with index 1. After its result call it "
        "with index 2. Then answer exactly MVP10-DIFFERENT-67.",
        tools=[COLLECT],
        max_rounds=3,
        expected_tools=[
            ["collect_evidence", 1],
            ["collect_evidence", 2],
        ],
    ),
    "failed_tool_repeated": scenario(
        "MVP10-FAILED-71",
        "Call failing_probe with index 1. Its error is expected; repeat exactly "
        "the same failing_probe once. After the second error answer exactly "
        "MVP10-FAILED-71.",
        tools=[FAIL],
        max_rounds=3,
        expected_tools=[["failing_probe", 1], ["failing_probe", 1]],
    ),
    "repeat_then_protected_close": scenario(
        "MVP10-CLOSE-83",
        "A test injector will call collect_evidence twice with index 1. When "
        "the operation budget requires finalization, answer exactly "
        "MVP10-CLOSE-83 without another tool.",
        tools=[COLLECT],
        max_rounds=2,
        expected_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        injected=[("collect_evidence", 1), ("collect_evidence", 1)],
        expected_guard=True,
        expected_partial=True,
        parity=False,
        max_tool_calls=2,
    ),
}


def openai_tool_response(name, index, call_id):
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps({"index": index}),
                    },
                }],
            },
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


class TracingLlm:
    def __init__(self, delegate, injected):
        self.delegate = delegate
        self.injected = list(injected)
        self.inferences = 0
        self.real_model_posts = 0
        self.tool_batches = []
        self.warning_inferences = 0
        self.warning_messages = []
        self.responses = []

    def _observe(self, messages):
        warnings = [
            item.get("content", "")
            for item in messages
            if item.get("role") == "system"
            and item.get("content", "").startswith(REPEAT_WARNING)
        ]
        if warnings:
            self.warning_inferences += 1
            self.warning_messages.extend(warnings)

    def chat_with_tools(self, messages, *args, **kwargs):
        self.inferences += 1
        self._observe(messages)
        if self.injected:
            name, index = self.injected.pop(0)
            response = openai_tool_response(
                name,
                index,
                f"mvp10-injected-{self.inferences}",
            )
            with patch("services.llm_service._post", return_value=response):
                value = self.delegate.chat_with_tools(messages, *args, **kwargs)
        else:
            self.real_model_posts += 1
            value = self.delegate.chat_with_tools(messages, *args, **kwargs)
        calls = value.get("tool_calls") or []
        self.tool_batches.append(len(calls))
        self.responses.append(copy.deepcopy(value))
        return value

    def chat(self, messages, *args, **kwargs):
        self.inferences += 1
        self.real_model_posts += 1
        self._observe(messages)
        value = self.delegate.chat(messages, *args, **kwargs)
        self.responses.append({"content": value})
        return value


def request(backend_url, method, path, body=None):
    value = urllib.request.Request(
        f"{backend_url}{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Workspace-Id": "mvp10-benchmark",
        },
    )
    with urllib.request.urlopen(value, timeout=30) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


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


def create_emitter(backend_url, assistant_id):
    created = request(
        backend_url,
        "POST",
        f"/assistants/{assistant_id}/messages",
        {"content": "MVP 10 exact tool repeat benchmark probe"},
    )
    return ExecutionEmitter(activate_execution(created["executionId"]))


def guard_signals(events):
    return [
        (event.get("payload") or {}).get("loopGuardSignal")
        for event in events
        if event.get("eventType") == "progress.reported"
        and (event.get("payload") or {}).get("kind") == "loop_guard_triggered"
    ]


def warning_starts(events):
    return [
        event
        for event in events
        if event.get("eventType") == "operation.started"
        and (event.get("payload") or {}).get("operationKind") == "inference"
        and (event.get("payload") or {}).get("loopGuardWarningApplied") is True
    ]


def budget_usage(progress):
    grants = (((progress.get("ledger") or {}).get("operationBudget") or {})
              .get("grants", {}))
    if len(grants) != 1:
        return None
    usage = next(iter(grants.values())).get("usage") or {}
    return {
        bucket: int((usage.get(bucket) or {}).get("consumed", 0))
        for bucket in ("normal", "repair", "closing", "tool")
    }


def normalized_calls(calls):
    return [[name, index] for name, index in calls]


def expected_calls(item, enabled):
    expected = list(item["expectedTools"])
    if not item["expectedGuard"] or item["expectedPartial"]:
        return expected
    if item["expectedPostWarning"] == "answer":
        return expected if enabled else [*expected, ["collect_evidence", 1]]
    if item["expectedPostWarning"] == "changed_arguments":
        return expected if enabled else [
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["collect_evidence", 1],
        ]
    return expected


def run_sample(service, backend_url, assistant_id, name, enabled):
    item = SCENARIOS[name]
    emitter = create_emitter(backend_url, assistant_id)
    token = activate_emitter(emitter)
    llm = TracingLlm(service, item["injected"])
    calls = []
    runtime_tools = {
        "collect_evidence": Tool(
            COLLECT,
            lambda *_args: {},
            lambda result: (f"Evidence {result['index']} collected", None),
        ),
        "inspect_evidence": Tool(
            INSPECT,
            lambda *_args: {},
            lambda result: (f"Evidence {result['index']} inspected", None),
        ),
        "failing_probe": Tool(FAIL, lambda *_args: {}),
    }
    spec = AgentSpec(
        name="mvp10-benchmark",
        config_key="mvp10-benchmark",
        system_prompt=item["system"],
        tool_names=frozenset(
            tool["function"]["name"] for tool in item["tools"]
        ),
        max_rounds=item["maxRounds"],
        fallback_max_tokens=128,
    )
    messages = [
        {"role": "system", "content": item["system"]},
        {"role": "user", "content": item["user"]},
    ]

    def dispatch(tool_name, arguments, _ctx):
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        index = int((parsed or {}).get("index") or 0)
        calls.append((tool_name, index))
        if tool_name == "failing_probe":
            return {"error": "expected_probe_failure", "index": index}
        return {
            "index": index,
            "validationMarker": item["marker"],
        }

    started = time.perf_counter()
    try:
        with patch(
            "agents.loop.get_task_config",
            return_value={
                "max_rounds": item["maxRounds"],
                "normal_inference_soft_limit": 0,
                "max_tokens": 128,
                "max_tool_calls": item["maxToolCalls"],
                "tool_call_soft_limit": 0,
                "exact_tool_repeat_warning": enabled,
            },
        ), patch("agents.loop.get_llm_params", return_value={}), patch(
            "agents.loop.get_llm_service", return_value=llm
        ), patch.dict(REGISTRY, runtime_tools, clear=False):
            result = run_agent_loop(
                spec,
                messages,
                ToolContext(execution=emitter),
                item["tools"],
                dispatch,
            )
        if result.kind == "final_text":
            emitter.record_final_message(
                result.content or "",
                generation_source=result.completion_source or "model",
            )
            emitter.flush_evidence()
    finally:
        reset_emitter(token)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    context = emitter.context
    progress = request(
        backend_url,
        "GET",
        f"/executions/{context['rootExecutionId']}/progress",
    )
    events = request(
        backend_url,
        "GET",
        f"/executions/{context['rootExecutionId']}/events?limit=500",
    )["events"]
    prompts = materialized_prompts(context["rootExecutionId"])
    signals = guard_signals(events)
    starts = warning_starts(events)
    prompt_warning_count = sum(
        str(message.get("content") or "").startswith(REPEAT_WARNING)
        for prompt in prompts
        for message in prompt.get("messages", [])
        if isinstance(message, dict)
    )
    expected_signal_count = 1 if enabled and item["expectedGuard"] else 0
    expected_warning_count = (
        1 if enabled and item["expectedWarning"] else 0
    )
    content = result.content or ""
    partial_matches = (
        result.kind == "final_text"
        and result.completion_kind == "partial"
        and item["marker"] in content
        if item["expectedPartial"]
        else result.kind == "final_text"
        and result.completion_kind is None
        and (
            bool(content.strip())
            if item["expectedGuard"] and not enabled
            else item["marker"] in content
        )
    )
    actual_calls = normalized_calls(calls)
    calls_match = (
        actual_calls[:len(item["expectedTools"])] == item["expectedTools"]
        if item["expectedGuard"] and not enabled
        else actual_calls == expected_calls(item, enabled)
    )
    guard_state = ((progress.get("ledger") or {}).get("loopGuards") or {})
    guard = next(iter(guard_state.values()), {}).get("exactToolRepeat", {})
    warning_pending_expected = bool(
        enabled and item["expectedGuard"] and not item["expectedWarning"]
    )
    correct = (
        partial_matches
        and calls_match
        and len(signals) == expected_signal_count
        and len(starts) == expected_warning_count
        and prompt_warning_count == expected_warning_count
        and bool(guard.get("warningPending", False)) == warning_pending_expected
    )
    return {
        "mode": "mvp10" if enabled else "mvp09_control",
        "elapsedMs": elapsed_ms,
        "realModelPosts": llm.real_model_posts,
        "inferences": llm.inferences,
        "toolBatches": llm.tool_batches,
        "tools": actual_calls,
        "expectedTools": expected_calls(item, enabled),
        "resultKind": result.kind,
        "completionKind": result.completion_kind,
        "completionReason": result.completion_reason,
        "content": content,
        "guardSignals": len(signals),
        "warningStarts": len(starts),
        "warningPrompts": prompt_warning_count,
        "warningInferences": llm.warning_inferences,
        "warningPending": bool(guard.get("warningPending", False)),
        "budgetUsage": budget_usage(progress),
        "rootExecutionId": context["rootExecutionId"],
        "correct": correct,
    }


def collect(service, backend_url, assistant_id, name, samples, max_attempts):
    values = []
    accepted = {"mvp09_control": 0, "mvp10": 0}
    for attempt in range(max_attempts):
        order = (False, True) if attempt % 2 == 0 else (True, False)
        for enabled in order:
            mode = "mvp10" if enabled else "mvp09_control"
            if accepted[mode] >= samples:
                continue
            value = run_sample(service, backend_url, assistant_id, name, enabled)
            values.append(value)
            print(name, json.dumps(value, ensure_ascii=False), flush=True)
            if value["correct"]:
                accepted[mode] += 1
        if all(count >= samples for count in accepted.values()):
            break
    if any(count < samples for count in accepted.values()):
        raise RuntimeError(f"{name}: insufficient correct samples: {accepted}")
    return values


def accepted(values, mode, key, samples):
    return [
        value[key]
        for value in values
        if value["correct"] and value["mode"] == mode
    ][:samples]


def summarize(values, name, samples):
    item = SCENARIOS[name]
    control = accepted(values, "mvp09_control", "elapsedMs", samples)
    current = accepted(values, "mvp10", "elapsedMs", samples)
    control_median = statistics.median(control)
    current_median = statistics.median(current)
    delta = current_median - control_median
    threshold = max(round(control_median * 0.10), 150)
    parity = item["parity"]
    mvp_tools = accepted(values, "mvp10", "tools", samples)
    post_warning = item["expectedPostWarning"]
    recovered = (
        samples
        if post_warning in {"answer", "changed_arguments"}
        else 0
    )
    persisted = samples if post_warning == "repeated" else 0
    return {
        "mvp09ControlMs": control,
        "mvp10Ms": current,
        "mvp09ControlMedianMs": control_median,
        "mvp10MedianMs": current_median,
        "deltaMs": delta,
        "thresholdMs": threshold,
        "latencyThresholdApplies": parity,
        "mvp10Tools": mvp_tools,
        "mvp10Inferences": accepted(values, "mvp10", "inferences", samples),
        "mvp10RealModelPosts": accepted(
            values, "mvp10", "realModelPosts", samples
        ),
        "mvp10BudgetUsage": accepted(values, "mvp10", "budgetUsage", samples),
        "mvp10GuardSignals": accepted(
            values, "mvp10", "guardSignals", samples
        ),
        "mvp10WarningPrompts": accepted(
            values, "mvp10", "warningPrompts", samples
        ),
        "recoveredAfterWarning": recovered,
        "persistedAfterWarning": persisted,
        "passed": not parity or delta <= threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ensure_ingest_token()
    backend_url = os.environ.get(
        "BACKEND_URL", "http://127.0.0.1:3000"
    ).rstrip("/")
    assistant = request(
        backend_url,
        "POST",
        "/assistants",
        {"name": "MVP 10 validation", "systemPrompt": "Validation fixture"},
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
    selected = (
        {args.scenario: SCENARIOS[args.scenario]}
        if args.scenario else SCENARIOS
    )
    raw = {
        name: collect(
            service,
            backend_url,
            assistant["id"],
            name,
            args.samples,
            args.max_attempts,
        )
        for name in selected
    }
    scenarios = {
        name: summarize(values, name, args.samples)
        for name, values in raw.items()
    }
    injected = [
        value
        for name, value in scenarios.items()
        if SCENARIOS[name]["expectedWarning"]
    ]
    recovered = sum(value["recoveredAfterWarning"] for value in injected)
    persisted = sum(value["persistedAfterWarning"] for value in injected)
    false_positive_control_detections = sum(
        value["guardSignals"]
        for values in raw.values()
        for value in values
        if value["correct"] and value["mode"] == "mvp09_control"
    )
    report = {
        "model": Path(params["model_path"]).name,
        "samplesPerMode": args.samples,
        "scenarios": scenarios,
        "quality": {
            "warningSamples": recovered + persisted,
            "recovered": recovered,
            "persisted": persisted,
            "recoveryPercent": round(
                100 * recovered / max(1, recovered + persisted), 1
            ),
            "falsePositiveControlDetections": false_positive_control_detections,
        },
        "passed": all(value["passed"] for value in scenarios.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
