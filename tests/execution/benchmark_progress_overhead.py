import argparse
import statistics
import time
import uuid
from pathlib import Path
from unittest.mock import patch

from agents.loop import run_agent_loop
from lib.execution import ExecutionEmitter, activate_emitter, reset_emitter
from lib.framework.agent import AgentSpec
from lib.framework.tool import ToolContext
from tests.execution.bench_harness import (
    accepted,
    backend_client,
    collect_paired,
    deterministic_service,
    resolve_backend_url,
    write_report,
)

WORKSPACE = "mvp04-benchmark"
request = backend_client(WORKSPACE, timeout=10)


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
    "search_fixture": {
        "type": "function",
        "function": {
            "name": "search_fixture",
            "description": "Find the path containing the validation value.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "read_fixture": {
        "type": "function",
        "function": {
            "name": "read_fixture",
            "description": "Read a fixture path returned by search_fixture.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    "continue_probe": {
        "type": "function",
        "function": {
            "name": "continue_probe",
            "description": "Record one validation step before finalization.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
}


SCENARIOS = {
    "direct_with_catalog": {
        "marker": "MVP04-DIRECT-17",
        "tools": ["lookup_value"],
        "max_rounds": 3,
        "system": (
            "/no_think\nAnswer exactly MVP04-DIRECT-17. Do not call any tool; "
            "the catalog is intentionally irrelevant."
        ),
        "user": "Return the exact validation marker.",
        "expected_tools": [],
    },
    "one_tool": {
        "marker": "MVP04-ONE-23",
        "tools": ["lookup_value"],
        "max_rounds": 3,
        "system": (
            "/no_think\nCall lookup_value exactly once with key alpha. After its "
            "result, answer with only the returned value."
        ),
        "user": "Resolve the validation value using the tool.",
        "expected_tools": ["lookup_value"],
    },
    "two_tools": {
        "marker": "MVP04-TWO-31",
        "tools": ["search_fixture", "read_fixture"],
        "max_rounds": 4,
        "system": (
            "/no_think\nFirst call search_fixture with query validation. Then call "
            "read_fixture with the returned path. Do not skip or repeat a step. "
            "Finally answer with only the value returned by read_fixture."
        ),
        "user": "Find and read the validation value.",
        "expected_tools": ["search_fixture", "read_fixture"],
    },
    "output_repair": {
        "marker": "MVP04-REPAIR-47",
        "tools": ["lookup_value"],
        "max_rounds": 3,
        "system": (
            "/no_think\nDo not call a tool. Answer exactly MVP04-REPAIR-47 when "
            "asked to continue after an invalid empty response."
        ),
        "user": "Return the repair validation marker.",
        "expected_tools": [],
        "inject_empty": True,
    },
    "forced_finalization": {
        "marker": "MVP04-FORCED-59",
        "tools": ["continue_probe"],
        "max_rounds": 1,
        "system": (
            "/no_think\nOn the first request call continue_probe exactly once. If "
            "later required to answer without tools, answer exactly MVP04-FORCED-59."
        ),
        "user": "Perform the validation step, then finish.",
        "expected_tools": ["continue_probe"],
    },
}


class CountingLlm:
    def __init__(self, delegate, inject_empty=False):
        self.delegate = delegate
        self.inject_empty = inject_empty
        self.inferences = 0

    def chat_with_tools(self, *args, **kwargs):
        self.inferences += 1
        response = self.delegate.chat_with_tools(*args, **kwargs)
        if self.inject_empty:
            self.inject_empty = False
            return {"content": "", "tool_calls": []}
        return response

    def chat(self, *args, **kwargs):
        self.inferences += 1
        return self.delegate.chat(*args, **kwargs)


def create_emitter(backend_url):
    execution = request(
        backend_url,
        "POST",
        "/executions",
        {"taskType": "search", "content": "MVP 04 benchmark probe"},
    )
    context = {
        "rootExecutionId": execution["rootExecutionId"],
        "executionId": execution["executionId"],
        "attemptId": str(uuid.uuid4()),
        "causedByEventId": execution["lastEventId"],
    }
    return ExecutionEmitter(context)


def dispatch_for(calls, name, _arguments, _ctx):
    calls.append(name)
    if name == "lookup_value":
        return {"value": "MVP04-ONE-23"}
    if name == "search_fixture":
        return {"path": "fixture://mvp04"}
    if name == "read_fixture":
        return {"value": "MVP04-TWO-31"}
    if name == "continue_probe":
        return {"step": "recorded"}
    return {"error": "unknown_tool"}


def run_sample(service, backend_url, name, instrumented):
    scenario = SCENARIOS[name]
    emitter = create_emitter(backend_url) if instrumented else None
    token = activate_emitter(emitter) if emitter else None
    llm = CountingLlm(service, bool(scenario.get("inject_empty")))
    calls = []
    spec = AgentSpec(
        name="mvp04-benchmark",
        config_key="mvp04-benchmark",
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
        if emitter and result.kind == "final_text":
            emitter.record_final_message(result.content or "")
            emitter.flush_evidence()
    finally:
        if token is not None:
            reset_emitter(token)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    content = result.content or ""
    expected_marker = scenario["marker"]
    content_matches = expected_marker in content
    if name == "forced_finalization":
        content_matches = bool(content.strip())
    correct = (
        result.kind == "final_text"
        and content_matches
        and calls == scenario["expected_tools"]
    )
    if name == "output_repair":
        correct = correct and llm.inferences >= 2
    if name == "forced_finalization":
        correct = correct and llm.inferences == 2
    return {
        "mode": "instrumented" if instrumented else "baseline",
        "elapsedMs": elapsed_ms,
        "instrumentationMs": emitter.instrumentation_ms if emitter else 0,
        "inferences": llm.inferences,
        "tools": calls,
        "resultKind": result.kind,
        "content": content,
        "correct": correct,
    }


def collect_scenario_samples(
    service,
    backend_url,
    name,
    sample_count,
    max_attempts,
):
    return collect_paired(
        lambda instrumented: run_sample(
            service, backend_url, name, instrumented
        ),
        name,
        sample_count,
        max_attempts,
        ("baseline", "instrumented"),
    )


def summarize_scenario(attempts, sample_count):
    baseline = accepted(attempts, "baseline", "elapsedMs", sample_count)
    instrumented = accepted(
        attempts,
        "instrumented",
        "elapsedMs",
        sample_count,
    )
    instrumentation = accepted(
        attempts,
        "instrumented",
        "instrumentationMs",
        sample_count,
    )
    baseline_outputs = accepted(
        attempts,
        "baseline",
        "content",
        sample_count,
    )
    instrumented_outputs = accepted(
        attempts,
        "instrumented",
        "content",
        sample_count,
    )
    baseline_median = statistics.median(baseline)
    instrumented_median = statistics.median(instrumented)
    delta = instrumented_median - baseline_median
    threshold = max(round(baseline_median * 0.10), 150)
    semantic_match = set(baseline_outputs) == set(instrumented_outputs)
    return {
        "baselineMs": baseline,
        "instrumentedMs": instrumented,
        "instrumentationMs": instrumentation,
        "baselineOutputs": baseline_outputs,
        "instrumentedOutputs": instrumented_outputs,
        "semanticMatch": semantic_match,
        "baselineMedianMs": baseline_median,
        "instrumentedMedianMs": instrumented_median,
        "deltaMs": delta,
        "thresholdMs": threshold,
        "passed": delta <= threshold and semantic_match,
    }


def build_report(results, model, sample_count):
    scenarios = {
        name: summarize_scenario(attempts, sample_count)
        for name, attempts in results.items()
    }
    return {
        "model": model,
        "samplesPerMode": sample_count,
        "scenarios": scenarios,
        "passed": all(value["passed"] for value in scenarios.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend_url = resolve_backend_url()
    service, params = deterministic_service()

    results = {}
    for name in SCENARIOS:
        results[name] = collect_scenario_samples(
            service,
            backend_url,
            name,
            args.samples,
            args.max_attempts,
        )

    report = build_report(
        results,
        Path(params["model_path"]).name,
        args.samples,
    )
    write_report(args.output, report)


if __name__ == "__main__":
    main()
