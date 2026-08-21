import argparse
import time
from pathlib import Path
from unittest.mock import patch

from agents.loop import run_agent_loop
from lib.execution import ExecutionEmitter, activate_emitter, reset_emitter
from lib.framework.agent import AgentSpec
from lib.framework.tool import ToolContext
from tests.execution.bench_harness import (
    accepted,
    activate_execution,
    backend_client,
    collect_paired,
    create_assistant,
    deterministic_service,
    latency_block,
    resolve_backend_url,
    write_report,
)
from tests.execution.benchmark_progress_overhead import (
    SCENARIOS as MVP04_SCENARIOS,
    TOOLS,
    CountingLlm,
    dispatch_for,
)

WORKSPACE = "mvp05-benchmark"
request = backend_client(WORKSPACE, timeout=10)


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
    ledger = progress.get("ledger") or {}
    budget = ledger.get("operationBudget") or ledger.get("inferenceBudget") or {}
    grants = budget.get("grants", {})
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
    return collect_paired(
        lambda budget_protocol: run_sample(
            service,
            backend_url,
            assistant_id,
            name,
            budget_protocol,
        ),
        name,
        sample_count,
        max_attempts,
        ("mvp04_control", "mvp05"),
    )


def summarize(attempts, name, sample_count):
    common = latency_block(
        attempts,
        sample_count,
        control_mode="mvp04_control",
        current_mode="mvp05",
        control_key="mvp04Control",
        current_key="mvp05",
    )
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
        **common,
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
        "passed": (
            common["deltaMs"] <= common["thresholdMs"]
            and semantic_match
            and trajectory_match
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend_url = resolve_backend_url()
    assistant = create_assistant(request, backend_url, "MVP 05 validation")
    service, params = deterministic_service()

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
    write_report(args.output, report)


if __name__ == "__main__":
    main()
