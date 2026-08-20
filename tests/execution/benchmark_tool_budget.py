import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

from agents.loop import run_agent_loop
from lib.execution import ExecutionEmitter, activate_emitter, reset_emitter
from lib.framework.agent import AgentSpec
from lib.framework.tool import ToolContext
from lib.llm.config import get_llm_params
from services.llm_service import get_llm_service
from tests.execution.benchmark_inference_budget import activate_execution
from tests.execution.benchmark_progress_overhead import CountingLlm


TOOLS = {
    "lookup_value": {
        "type": "function",
        "function": {
            "name": "lookup_value",
            "description": "Return the validation value for a key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    "first_value": {
        "type": "function",
        "function": {
            "name": "first_value",
            "description": "Return the first independent validation value.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "second_value": {
        "type": "function",
        "function": {
            "name": "second_value",
            "description": "Return the second independent validation value.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "failing_probe": {
        "type": "function",
        "function": {
            "name": "failing_probe",
            "description": "Return the expected validation error.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
}


SCENARIOS = {
    "direct_with_catalog": {
        "marker": "MVP06-DIRECT-17",
        "tools": ["lookup_value"],
        "max_rounds": 3,
        "system": (
            "/no_think\nAnswer exactly MVP06-DIRECT-17. Do not call any tool; "
            "the catalog is intentionally irrelevant."
        ),
        "user": "Return the exact validation marker.",
        "expected": [],
    },
    "one_tool": {
        "marker": "MVP06-ONE-23",
        "tools": ["lookup_value"],
        "max_rounds": 3,
        "system": (
            "/no_think\nCall lookup_value exactly once with key alpha. After its "
            "result, answer with only the returned value."
        ),
        "user": "Resolve the validation value using the tool.",
        "expected": ["lookup_value"],
    },
    "two_tools_one_batch": {
        "marker": "MVP06-BATCH-31",
        "tools": ["first_value", "second_value"],
        "max_rounds": 3,
        "system": (
            "/no_think\nOn the first turn call first_value and second_value "
            "together in the same assistant response. They are independent: do "
            "not wait for one result before calling the other. After both results, "
            "answer exactly MVP06-BATCH-31."
        ),
        "user": "Run both independent validation tools in one batch, then finish.",
        "expected": ["first_value", "second_value"],
        "requires_batch": True,
    },
    "tool_failure": {
        "marker": "MVP06-ERROR-47",
        "tools": ["failing_probe"],
        "max_rounds": 3,
        "system": (
            "/no_think\nCall failing_probe exactly once. Its error is expected; "
            "do not retry it. After receiving the error, answer exactly "
            "MVP06-ERROR-47."
        ),
        "user": "Run the expected failure probe and recover with the marker.",
        "expected": ["failing_probe"],
    },
    "tool_limit": {
        "marker": "MVP06-LIMIT-59",
        "tools": ["first_value", "second_value"],
        "max_rounds": 3,
        "system": (
            "/no_think\nOn the first turn call first_value and second_value "
            "together in the same assistant response. They are independent. If "
            "one call is skipped because the tool budget is exhausted, do not "
            "request more tools and answer exactly MVP06-LIMIT-59. If both return, "
            "also answer exactly MVP06-LIMIT-59."
        ),
        "user": "Run both independent validation tools in one batch, then finish.",
        "expected": ["first_value", "second_value"],
        "governed_expected": ["first_value"],
        "requires_batch": True,
        "governed_tool_limit": 1,
    },
}


EXPECTED_USAGE = {
    "direct_with_catalog": {"normal": 1, "repair": 0, "closing": 0, "tool": 0},
    "one_tool": {"normal": 2, "repair": 0, "closing": 0, "tool": 1},
    "two_tools_one_batch": {"normal": 2, "repair": 0, "closing": 0, "tool": 2},
    "tool_failure": {"normal": 2, "repair": 0, "closing": 0, "tool": 1},
    "tool_limit": {"normal": 1, "repair": 0, "closing": 1, "tool": 1},
}


class MeasuredEmitter(ExecutionEmitter):
    def __init__(self, context):
        super().__init__(context)
        self.tool_reservation_ms = 0

    def reserve_operation_budget(self, **kwargs):
        started = time.perf_counter()
        try:
            return super().reserve_operation_budget(**kwargs)
        finally:
            if kwargs.get("operation_kind") == "tool_call":
                self.tool_reservation_ms += round(
                    (time.perf_counter() - started) * 1000
                )


class Mvp05ControlEmitter(MeasuredEmitter):
    def start_tool(self, name, arguments, provider_tool_call_id, metadata=None):
        trace = dict(metadata or {})
        trace.pop("budgetGrantId", None)
        trace["loopKind"] = "synchronous_subagent"
        return super().start_tool(
            name,
            arguments,
            provider_tool_call_id,
            trace,
        )


class BatchCountingLlm(CountingLlm):
    def __init__(self, delegate):
        super().__init__(delegate)
        self.tool_batches = []

    def chat_with_tools(self, *args, **kwargs):
        response = super().chat_with_tools(*args, **kwargs)
        self.tool_batches.append(len(response.get("tool_calls") or []))
        return response


def request(backend_url, method, path, body=None):
    value = urllib.request.Request(
        f"{backend_url}{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Workspace-Id": "mvp06-benchmark",
        },
    )
    with urllib.request.urlopen(value, timeout=10) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def create_emitter(backend_url, assistant_id, governed):
    created = request(
        backend_url,
        "POST",
        f"/assistants/{assistant_id}/messages",
        {"content": "MVP 06 tool budget benchmark probe"},
    )
    context = activate_execution(created["executionId"])
    emitter_type = MeasuredEmitter if governed else Mvp05ControlEmitter
    return emitter_type(context), context


def dispatch_for(calls, name, _arguments, _ctx):
    calls.append(name)
    if name == "lookup_value":
        return {"value": "MVP06-ONE-23"}
    if name == "first_value":
        return {"value": "first"}
    if name == "second_value":
        return {"value": "second"}
    if name == "failing_probe":
        return {"error": "expected_validation_failure"}
    return {"error": "unknown_tool"}


def budget_usage(progress):
    budget = ((progress.get("ledger") or {}).get("operationBudget") or {})
    grants = budget.get("grants", {})
    if len(grants) != 1:
        return None
    usage = next(iter(grants.values())).get("usage") or {}
    return {
        bucket: int((usage.get(bucket) or {}).get("consumed", 0))
        for bucket in ("normal", "repair", "closing", "tool")
    }


def run_sample(service, backend_url, assistant_id, name, governed):
    scenario = SCENARIOS[name]
    emitter, context = create_emitter(backend_url, assistant_id, governed)
    token = activate_emitter(emitter)
    llm = BatchCountingLlm(service)
    calls = []
    requested_tool_limit = (
        int(scenario.get("governed_tool_limit", 6)) if governed else 0
    )
    spec = AgentSpec(
        name="mvp06-benchmark",
        config_key="mvp06-benchmark",
        system_prompt=scenario["system"],
        tool_names=frozenset(scenario["tools"]),
        max_rounds=scenario["max_rounds"],
        fallback_max_tokens=96,
    )
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
                "max_tool_calls": requested_tool_limit,
            },
        ), patch("agents.loop.get_llm_params", return_value={}), patch(
            "agents.loop.get_llm_service", return_value=llm
        ):
            result = run_agent_loop(
                spec,
                messages,
                ToolContext(execution=emitter),
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
    expected_calls = (
        scenario.get("governed_expected", scenario["expected"])
        if governed
        else scenario["expected"]
    )
    expected_partial = governed and name == "tool_limit"
    classification_matches = (
        result.completion_kind == "partial"
        and result.completion_reason == "tool_budget_exhausted"
        if expected_partial
        else result.completion_kind is None and result.completion_reason is None
    )
    batch_matches = (
        not scenario.get("requires_batch")
        or bool(llm.tool_batches)
        and llm.tool_batches[0] == 2
    )
    progress = request(
        backend_url,
        "GET",
        f"/executions/{context['rootExecutionId']}/progress",
    )
    usage = budget_usage(progress)
    expected_usage = (
        EXPECTED_USAGE[name]
        if governed
        else {
            **EXPECTED_USAGE[name],
            "tool": 0,
            **(
                {"normal": 2, "closing": 0}
                if name == "tool_limit"
                else {}
            ),
        }
    )
    correct = (
        result.kind == "final_text"
        and scenario["marker"] in content
        and calls == expected_calls
        and classification_matches
        and batch_matches
        and usage == expected_usage
    )
    return {
        "mode": "mvp06" if governed else "mvp05_control",
        "elapsedMs": elapsed_ms,
        "instrumentationMs": emitter.instrumentation_ms,
        "toolReservationMs": emitter.tool_reservation_ms,
        "inferences": llm.inferences,
        "toolBatches": llm.tool_batches,
        "tools": calls,
        "resultKind": result.kind,
        "completionKind": result.completion_kind,
        "completionReason": result.completion_reason,
        "content": content,
        "budgetUsage": usage,
        "expectedBudgetUsage": expected_usage,
        "correct": correct,
    }


def collect_samples(service, backend_url, assistant_id, name, samples, attempts):
    values = []
    accepted = {"mvp05_control": 0, "mvp06": 0}
    for attempt in range(attempts):
        order = (False, True) if attempt % 2 == 0 else (True, False)
        for governed in order:
            mode = "mvp06" if governed else "mvp05_control"
            if accepted[mode] >= samples:
                continue
            sample = run_sample(service, backend_url, assistant_id, name, governed)
            values.append(sample)
            print(name, json.dumps(sample, ensure_ascii=False), flush=True)
            if sample["correct"]:
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
    control = accepted(values, "mvp05_control", "elapsedMs", samples)
    governed = accepted(values, "mvp06", "elapsedMs", samples)
    control_median = statistics.median(control)
    governed_median = statistics.median(governed)
    delta = governed_median - control_median
    threshold = max(round(control_median * 0.10), 150)
    semantic_match = set(accepted(
        values, "mvp05_control", "content", samples
    )) == set(accepted(values, "mvp06", "content", samples))
    expected_divergence = name == "tool_limit"
    trajectory_match = (
        accepted(values, "mvp05_control", "inferences", samples)
        == accepted(values, "mvp06", "inferences", samples)
        and accepted(values, "mvp05_control", "tools", samples)
        == accepted(values, "mvp06", "tools", samples)
    )
    passed = semantic_match and (
        expected_divergence or (trajectory_match and delta <= threshold)
    )
    return {
        "mvp05ControlMs": control,
        "mvp06Ms": governed,
        "mvp05ControlMedianMs": control_median,
        "mvp06MedianMs": governed_median,
        "deltaMs": delta,
        "thresholdMs": threshold,
        "latencyThresholdApplies": not expected_divergence,
        "semanticMatch": semantic_match,
        "trajectoryMatch": trajectory_match,
        "mvp06ToolReservationMs": accepted(
            values, "mvp06", "toolReservationMs", samples
        ),
        "mvp06Inferences": accepted(values, "mvp06", "inferences", samples),
        "mvp06Tools": accepted(values, "mvp06", "tools", samples),
        "mvp06CompletionKinds": accepted(
            values, "mvp06", "completionKind", samples
        ),
        "mvp06CompletionReasons": accepted(
            values, "mvp06", "completionReason", samples
        ),
        "mvp06BudgetUsage": accepted(
            values, "mvp06", "budgetUsage", samples
        ),
        "passed": passed,
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
        {"name": "MVP 06 validation", "systemPrompt": "Validation fixture"},
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

    raw = {
        name: collect_samples(
            service,
            backend_url,
            assistant["id"],
            name,
            args.samples,
            args.max_attempts,
        )
        for name in SCENARIOS
    }
    scenarios = {
        name: summarize(values, name, args.samples)
        for name, values in raw.items()
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
