import argparse
import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

from agents.loop import run_agent_loop
from lib.execution import ExecutionEmitter, activate_emitter, reset_emitter
from lib.framework.agent import AgentSpec
from lib.framework.tool import ToolContext
from services import llm_service as llm_module
from tests.execution.bench_harness import (
    accepted,
    activate_execution,
    backend_client,
    collect_paired,
    create_assistant,
    deterministic_service,
    latency_block,
    materialized_prompts,
    resolve_backend_url,
    write_report,
)

WORKSPACE = "mvp08-benchmark"
request = backend_client(WORKSPACE)


NORMAL_WARNING_PREFIX = "Normal inference budget is low: 1 of 3 calls remain."
TOOL_WARNING_PREFIX = "Tool budget is low: 2 of 3 calls remain."
COLLECT_EVIDENCE = {
    "type": "function",
    "function": {
        "name": "collect_evidence",
        "description": "Collect one required validation evidence item by index.",
        "parameters": {
            "type": "object",
            "properties": {"index": {"type": "integer", "minimum": 1}},
            "required": ["index"],
        },
    },
}
COLLECT_INITIAL = {
    "type": "function",
    "function": {
        "name": "collect_initial",
        "description": "Collect the first required item and its authorization code.",
        "parameters": {
            "type": "object",
            "properties": {"index": {"type": "integer", "const": 1}},
            "required": ["index"],
        },
    },
}
COLLECT_FOLLOWUP = {
    "type": "function",
    "function": {
        "name": "collect_followup",
        "description": (
            "Collect the next required item. Pass the authorization code returned "
            "by the preceding item."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 2},
                "authorizationCode": {"type": "string"},
            },
            "required": ["index", "authorizationCode"],
        },
    },
}


def scenario(
    marker,
    instruction,
    batches,
    tools,
    inferences,
    usage,
    *,
    offered="evidence",
    tool_soft_limit=0,
    partial=False,
):
    return {
        "marker": marker,
        "system": f"/no_think\n{instruction}",
        "user": "Complete the validation sequence exactly as instructed.",
        "batches": batches,
        "tools": tools,
        "inferences": inferences,
        "usage": usage,
        "offered": offered,
        "toolSoftLimit": tool_soft_limit,
        "partial": partial,
    }


SCENARIOS = {
    "direct": scenario(
        "MVP08-DIRECT-11",
        "Answer exactly MVP08-DIRECT-11. Do not call collect_evidence.",
        [0],
        [],
        1,
        {"normal": 1, "repair": 0, "closing": 0, "tool": 0},
        offered="none",
    ),
    "one_tool_then_answer": scenario(
        "MVP08-ONE-23",
        "Call collect_evidence once with index 1. After its result, answer "
        "exactly MVP08-ONE-23 without another tool.",
        [1, 0],
        [1],
        2,
        {"normal": 2, "repair": 0, "closing": 0, "tool": 1},
    ),
    "finish_with_existing_evidence": scenario(
        "MVP08-EXISTING-31",
        "In the first response call collect_evidence twice together, with "
        "indices 1 and 2. Those results are sufficient. After both results, "
        "answer exactly MVP08-EXISTING-31 without another tool.",
        [2, 0],
        [1, 2],
        2,
        {"normal": 2, "repair": 0, "closing": 0, "tool": 2},
    ),
    "required_third_inference": scenario(
        "MVP08-THIRD-43",
        "First call collect_initial with index 1. Its result provides the code "
        "required to call collect_followup with index 2. That follow-up is "
        "required. After index 2 returns, answer exactly MVP08-THIRD-43.",
        [1, 1, 0],
        [1, 2],
        3,
        {"normal": 3, "repair": 0, "closing": 0, "tool": 2},
        offered="chain",
    ),
    "both_soft_limits": scenario(
        "MVP08-BOTH-53",
        "Call collect_evidence once with index 1. After its result, answer "
        "exactly MVP08-BOTH-53 without another tool.",
        [1, 0],
        [1],
        2,
        {"normal": 2, "repair": 0, "closing": 0, "tool": 1},
        tool_soft_limit=1,
    ),
    "normal_limit_then_closing": scenario(
        "MVP08-CLOSING-67",
        "First call collect_initial with index 1. Then call collect_followup "
        "sequentially with indices 2, 3, and 4, always using the code returned "
        "by the preceding item. Every follow-up is required. If the normal "
        "inference budget produces a request without tool definitions before "
        "index 4, do not write a tool call or XML. In that request answer "
        "exactly MVP08-CLOSING-67.",
        [1, 1, 1],
        [1, 2, 3],
        4,
        {"normal": 3, "repair": 0, "closing": 1, "tool": 3},
        offered="chain",
        partial=True,
    ),
}


class MeasuredEmitter(ExecutionEmitter):
    def __init__(self, context):
        super().__init__(context)
        self.inference_reservation_ms = 0

    def reserve_operation_budget(self, **kwargs):
        started = time.perf_counter()
        try:
            return super().reserve_operation_budget(**kwargs)
        finally:
            if kwargs.get("operation_kind") == "inference":
                self.inference_reservation_ms += round(
                    (time.perf_counter() - started) * 1000
                )


class TracingLlm:
    def __init__(self, delegate):
        self.delegate = delegate
        self.inferences = 0
        self.tool_batches = []

    def chat_with_tools(self, messages, *args, **kwargs):
        self.inferences += 1
        response = self.delegate.chat_with_tools(messages, *args, **kwargs)
        self.tool_batches.append(len(response.get("tool_calls") or []))
        return response

    def chat(self, messages, *args, **kwargs):
        self.inferences += 1
        return self.delegate.chat(messages, *args, **kwargs)


def create_emitter(backend_url, assistant_id):
    created = request(
        backend_url,
        "POST",
        f"/assistants/{assistant_id}/messages",
        {"content": "MVP 08 normal inference soft limit benchmark probe"},
    )
    return MeasuredEmitter(activate_execution(created["executionId"]))


def budget_state(progress):
    grants = (
        ((progress.get("ledger") or {}).get("operationBudget") or {})
        .get("grants", {})
    )
    if len(grants) != 1:
        return None, {}, {}
    grant = next(iter(grants.values()))
    usage = grant.get("usage") or {}
    consumed = {
        bucket: int((usage.get(bucket) or {}).get("consumed", 0))
        for bucket in ("normal", "repair", "closing", "tool")
    }
    return consumed, usage.get("normal") or {}, usage.get("tool") or {}


def warning_positions(body, prefix):
    return [
        index
        for index, message in enumerate(body.get("messages") or [])
        if message.get("role") == "system"
        and str(message.get("content") or "").startswith(prefix)
    ]


def signal_count(events, operation_kind):
    return sum(
        event.get("eventType") == "progress.reported"
        and (event.get("payload") or {}).get("kind")
        == "budget_soft_limit_reached"
        and ((event.get("payload") or {}).get("signal") or {}).get(
            "operationKind"
        )
        == operation_kind
        for event in events
    )


def warning_started_count(events):
    return sum(
        event.get("eventType") == "operation.started"
        and (event.get("payload") or {}).get("operationKind") == "inference"
        and (event.get("payload") or {}).get("budgetBucket") == "normal"
        and (event.get("payload") or {}).get(
            "budgetSoftLimitWarningApplied"
        )
        is True
        for event in events
    )


def run_sample(service, backend_url, assistant_id, name, soft_enabled):
    item = SCENARIOS[name]
    offered_tools = {
        "none": [],
        "evidence": [COLLECT_EVIDENCE],
        "chain": [COLLECT_INITIAL, COLLECT_FOLLOWUP],
    }[item["offered"]]
    emitter = create_emitter(backend_url, assistant_id)
    token = activate_emitter(emitter)
    llm = TracingLlm(service)
    calls = []
    call_names = []
    dispatched_requests = []
    messages = [
        {"role": "system", "content": item["system"]},
        {"role": "user", "content": item["user"]},
    ]
    spec = AgentSpec(
        name="mvp08-benchmark",
        config_key="mvp08-benchmark",
        system_prompt=item["system"],
        tool_names=frozenset(
            tool["function"]["name"] for tool in offered_tools
        ),
        max_rounds=3,
        fallback_max_tokens=128,
    )

    def dispatch(tool_name, arguments, _ctx):
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        index = int(arguments.get("index") or 0)
        calls.append(index)
        call_names.append(tool_name)
        result = {"index": index, "validationMarker": item["marker"]}
        if tool_name in {"collect_initial", "collect_followup"}:
            result["authorizationCode"] = f"MVP08-AUTH-{index + 1}"
        return result

    original_post = llm_module._post

    def observe_post(url, payload, stream=False):
        if url.endswith("/v1/chat/completions") and not stream:
            dispatched_requests.append(copy.deepcopy(payload))
        return original_post(url, payload, stream=stream)

    started = time.perf_counter()
    try:
        with patch(
            "agents.loop.get_task_config",
            return_value={
                "max_rounds": 3,
                "normal_inference_soft_limit": 2 if soft_enabled else 0,
                "max_tokens": 128,
                "max_tool_calls": 3,
                "tool_call_soft_limit": item["toolSoftLimit"],
            },
        ), patch("agents.loop.get_llm_params", return_value={}), patch(
            "agents.loop.get_llm_service", return_value=llm
        ), patch("services.llm_service._post", side_effect=observe_post):
            result = run_agent_loop(
                spec,
                messages,
                ToolContext(execution=emitter),
                offered_tools,
                dispatch,
            )
        if result.kind == "final_text":
            emitter.record_final_message(result.content or "")
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
        f"/executions/{context['rootExecutionId']}/events?limit=200",
    )["events"]
    artifacts = materialized_prompts(context["rootExecutionId"])
    usage, normal_state, tool_state = budget_state(progress)
    normal_signals = signal_count(events, "inference")
    tool_signals = signal_count(events, "tool_call")
    warning_started = warning_started_count(events)
    normal_warning_positions = [
        warning_positions(body, "Normal inference budget is low:")
        for body in dispatched_requests
    ]
    tool_warning_positions = [
        warning_positions(body, "Tool budget is low:")
        for body in dispatched_requests
    ]
    normal_warning_inferences = sum(
        bool(positions) for positions in normal_warning_positions
    )
    tool_warning_inferences = sum(
        bool(positions) for positions in tool_warning_positions
    )
    reaches_normal_soft = item["usage"]["normal"] >= 2
    expected_normal_warning = int(soft_enabled and reaches_normal_soft)
    expected_tool_warning = int(item["toolSoftLimit"] > 0)
    expected_call_names = (
        ["collect_initial"]
        + ["collect_followup"] * (len(item["tools"]) - 1)
        if item["offered"] == "chain"
        else ["collect_evidence"] * len(item["tools"])
    )
    classification_matches = (
        result.completion_kind == "partial"
        and result.completion_reason == "budget_exhausted"
        if item["partial"]
        else result.completion_kind is None
        and result.completion_reason is None
    )
    normal_warning_text_matches = all(
        any(
            str(message.get("content") or "").startswith(
                NORMAL_WARNING_PREFIX
            )
            for message in body.get("messages") or []
        )
        for body, positions in zip(
            dispatched_requests, normal_warning_positions
        )
        if positions
    )
    tool_warning_text_matches = all(
        any(
            str(message.get("content") or "").startswith(TOOL_WARNING_PREFIX)
            for message in body.get("messages") or []
        )
        for body, positions in zip(dispatched_requests, tool_warning_positions)
        if positions
    )
    both_warning_order_matches = True
    if name == "both_soft_limits" and soft_enabled:
        both_warning_order_matches = (
            len(tool_warning_positions) > 1
            and bool(tool_warning_positions[1])
            and bool(normal_warning_positions[1])
            and tool_warning_positions[1][0] < normal_warning_positions[1][0]
        )
    warning_only_on_second = (
        expected_normal_warning == 0
        and normal_warning_inferences == 0
        or expected_normal_warning == 1
        and len(normal_warning_positions) >= 2
        and bool(normal_warning_positions[1])
        and not any(normal_warning_positions[:1])
        and not any(normal_warning_positions[2:])
    )
    durable_messages_clean = not any(
        str(message.get("content") or "").startswith(
            "Normal inference budget is low:"
        )
        for message in messages
    )
    artifacts_match_dispatch = artifacts == dispatched_requests
    correct = (
        result.kind == "final_text"
        and item["marker"] in (result.content or "")
        and calls == item["tools"]
        and call_names == expected_call_names
        and llm.tool_batches == item["batches"]
        and llm.inferences == item["inferences"]
        and usage == item["usage"]
        and normal_signals == expected_normal_warning
        and warning_started == expected_normal_warning
        and normal_warning_inferences == expected_normal_warning
        and tool_signals == expected_tool_warning
        and tool_warning_inferences == expected_tool_warning
        and normal_warning_text_matches
        and tool_warning_text_matches
        and warning_only_on_second
        and both_warning_order_matches
        and artifacts_match_dispatch
        and durable_messages_clean
        and int(normal_state.get("softLimit", 0))
        == (2 if soft_enabled else 0)
        and bool(normal_state.get("softLimitReached"))
        == bool(soft_enabled and reaches_normal_soft)
        and normal_state.get("softLimitWarningPending") is False
        and bool(tool_state.get("softLimitReached"))
        == bool(item["toolSoftLimit"] > 0)
        and classification_matches
    )
    return {
        "mode": "mvp08" if soft_enabled else "mvp07_control",
        "elapsedMs": elapsed_ms,
        "inferenceReservationMs": emitter.inference_reservation_ms,
        "inferences": llm.inferences,
        "toolBatches": llm.tool_batches,
        "tools": calls,
        "content": result.content or "",
        "completionKind": result.completion_kind,
        "completionReason": result.completion_reason,
        "budgetUsage": usage,
        "normalBudgetState": normal_state,
        "toolBudgetState": tool_state,
        "normalSoftLimitSignals": normal_signals,
        "toolSoftLimitSignals": tool_signals,
        "warningStartedEvents": warning_started,
        "normalWarningInferences": normal_warning_inferences,
        "toolWarningInferences": tool_warning_inferences,
        "normalWarningPositions": normal_warning_positions,
        "toolWarningPositions": tool_warning_positions,
        "artifactsMatchDispatch": artifacts_match_dispatch,
        "durableMessagesClean": durable_messages_clean,
        "correct": correct,
    }


def collect(service, backend_url, assistant_id, name, samples, max_attempts):
    return collect_paired(
        lambda soft_enabled: run_sample(
            service, backend_url, assistant_id, name, soft_enabled
        ),
        name,
        samples,
        max_attempts,
        ("mvp07_control", "mvp08"),
    )


def summarize(values, name, samples):
    common = latency_block(
        values,
        samples,
        control_mode="mvp07_control",
        current_mode="mvp08",
        control_key="mvp07Control",
        current_key="mvp08",
    )
    latency_applies = name == "direct"
    return {
        **common,
        "latencyThresholdApplies": latency_applies,
        "mvp08InferenceReservationMs": accepted(
            values, "mvp08", "inferenceReservationMs", samples
        ),
        "mvp08Inferences": accepted(values, "mvp08", "inferences", samples),
        "mvp08Tools": accepted(values, "mvp08", "tools", samples),
        "mvp08Warnings": accepted(
            values, "mvp08", "normalWarningInferences", samples
        ),
        "mvp08Signals": accepted(
            values, "mvp08", "normalSoftLimitSignals", samples
        ),
        "mvp08CompletionKinds": accepted(
            values, "mvp08", "completionKind", samples
        ),
        "mvp08CompletionReasons": accepted(
            values, "mvp08", "completionReason", samples
        ),
        "mvp08BudgetUsage": accepted(
            values, "mvp08", "budgetUsage", samples
        ),
        "passed": (
            not latency_applies
            or common["deltaMs"] <= common["thresholdMs"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend_url = resolve_backend_url()
    assistant = create_assistant(request, backend_url, "MVP 08 validation")
    service, params = deterministic_service()

    selected = (
        {args.scenario: SCENARIOS[args.scenario]}
        if args.scenario
        else SCENARIOS
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
    report = {
        "model": Path(params["model_path"]).name,
        "samplesPerMode": args.samples,
        "scenarios": scenarios,
        "passed": all(value["passed"] for value in scenarios.values()),
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
