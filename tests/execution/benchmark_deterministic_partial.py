import argparse
import copy
import json
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from agents.loop import run_agent_loop
from lib.execution import ExecutionEmitter, activate_emitter, reset_emitter
from lib.framework.agent import AgentSpec
from lib.framework.tool import Tool, ToolContext
from services import llm_service as llm_module
from tests.execution.bench_harness import (
    accepted,
    activate_execution,
    backend_client,
    collect_paired,
    create_assistant,
    deterministic_service,
    index_tool_schema,
    latency_block,
    materialized_prompts,
    resolve_backend_url,
    write_report,
)
from tools import REGISTRY

WORKSPACE = "mvp09-benchmark"
request = backend_client(WORKSPACE)


EMPTY_RESPONSE = {
    "choices": [{
        "finish_reason": "stop",
        "message": {"content": ""},
    }],
    "usage": {"prompt_tokens": 1, "completion_tokens": 0},
}


COLLECT_EVIDENCE = index_tool_schema(
    "collect_evidence",
    "Collect one required validation evidence item by index.",
)
FAILING_PROBE = index_tool_schema(
    "failing_probe",
    "Run the required validation probe, which is expected to fail.",
)


def scenario(
    marker,
    instruction,
    *,
    tools,
    max_rounds,
    expected_calls,
    expected_batches,
    expected_control,
    expected_current,
    inject_empty=False,
    deny_closing=False,
    max_tool_calls=4,
):
    return {
        "marker": marker,
        "system": f"/no_think\n{instruction}",
        "user": "Complete the validation sequence exactly as instructed.",
        "tools": tools,
        "maxRounds": max_rounds,
        "expectedCalls": expected_calls,
        "expectedBatches": expected_batches,
        "expectedControl": expected_control,
        "expectedCurrent": expected_current,
        "injectEmpty": inject_empty,
        "denyClosing": deny_closing,
        "maxToolCalls": max_tool_calls,
    }


SCENARIOS = {
    "direct": scenario(
        "MVP09-DIRECT-11",
        "Answer exactly MVP09-DIRECT-11.",
        tools=[],
        max_rounds=1,
        expected_calls=[],
        expected_batches=[0],
        expected_control="full",
        expected_current="full",
    ),
    "one_tool_then_answer": scenario(
        "MVP09-ONE-23",
        "Call collect_evidence once with index 1. After its result, answer "
        "exactly MVP09-ONE-23 without another tool.",
        tools=[COLLECT_EVIDENCE],
        max_rounds=2,
        expected_calls=[1],
        expected_batches=[1, 0],
        expected_control="full",
        expected_current="full",
    ),
    "valid_closing": scenario(
        "MVP09-CLOSING-31",
        "Call collect_evidence once with index 1. When the operation budget "
        "requires finalization, answer exactly MVP09-CLOSING-31.",
        tools=[COLLECT_EVIDENCE],
        max_rounds=1,
        expected_calls=[1],
        expected_batches=[1],
        expected_control="model_partial",
        expected_current="model_partial",
    ),
    "one_tool_empty_closing": scenario(
        "MVP09-EMPTY-ONE-43",
        "Call collect_evidence once with index 1.",
        tools=[COLLECT_EVIDENCE],
        max_rounds=1,
        expected_calls=[1],
        expected_batches=[1],
        expected_control="invalid",
        expected_current="runtime_partial",
        inject_empty=True,
    ),
    "multiple_tools_empty_closing": scenario(
        "MVP09-EMPTY-MULTI-53",
        "In one response call collect_evidence twice, with indices 1 and 2.",
        tools=[COLLECT_EVIDENCE],
        max_rounds=1,
        expected_calls=[1, 2],
        expected_batches=[2],
        expected_control="invalid",
        expected_current="runtime_partial",
        inject_empty=True,
    ),
    "tool_limit_closing_denied": scenario(
        "MVP09-DENIED-67",
        "In one response call collect_evidence twice, with indices 1 and 2.",
        tools=[COLLECT_EVIDENCE],
        max_rounds=2,
        expected_calls=[1],
        expected_batches=[2],
        expected_control="invalid",
        expected_current="runtime_partial",
        deny_closing=True,
        max_tool_calls=1,
    ),
    "empty_closing_without_eligible_tool": scenario(
        "MVP09-NO-VALUE-71",
        "Call failing_probe once with index 1.",
        tools=[FAILING_PROBE],
        max_rounds=1,
        expected_calls=[1],
        expected_batches=[1],
        expected_control="invalid",
        expected_current="invalid",
        inject_empty=True,
    ),
}


class BenchmarkEmitter(ExecutionEmitter):
    def __init__(self, context, deny_closing=False):
        super().__init__(context)
        self.deny_closing = deny_closing
        self.preconsumed_closing_operation_id = None

    def _post(self, suffix, body):
        try:
            return super()._post(suffix, body)
        except RuntimeError as error:
            if suffix == "artifacts":
                rejected = next(
                    (
                        artifact
                        for artifact in body.get("artifacts", [])
                        if artifact.get("artifactId") in str(error)
                    ),
                    None,
                )
                if rejected:
                    raise RuntimeError(
                        f"{error}; rejected artifact kind={rejected.get('kind')}"
                    ) from error
            raise

    def request_progress_grant(self, request):
        grant = super().request_progress_grant(request)
        if self.deny_closing and grant and not self.preconsumed_closing_operation_id:
            operation_id = str(uuid.uuid4())
            self.reserve_operation_budget(
                grant_id=grant["grantId"],
                loop_id=request["loopId"],
                operation_id=operation_id,
                operation_kind="inference",
                bucket="closing",
                phase="forced_finalization",
                round=1,
                name="forced_finalization",
            )
            self.preconsumed_closing_operation_id = operation_id
        return grant


class TracingLlm:
    def __init__(self, delegate, inject_empty, empty_post):
        self.delegate = delegate
        self.inject_empty = inject_empty
        self.empty_post = empty_post
        self.method_calls = 0
        self.tool_batches = []

    def chat_with_tools(self, messages, *args, **kwargs):
        self.method_calls += 1
        response = self.delegate.chat_with_tools(messages, *args, **kwargs)
        self.tool_batches.append(len(response.get("tool_calls") or []))
        return response

    def chat(self, messages, *args, **kwargs):
        self.method_calls += 1
        if not self.inject_empty:
            return self.delegate.chat(messages, *args, **kwargs)
        with patch("services.llm_service._post", side_effect=self.empty_post):
            return self.delegate.chat(messages, *args, **kwargs)


def create_emitter(backend_url, assistant_id, deny_closing):
    created = request(
        backend_url,
        "POST",
        f"/assistants/{assistant_id}/messages",
        {"content": "MVP 09 deterministic partial benchmark probe"},
    )
    return BenchmarkEmitter(
        activate_execution(created["executionId"]),
        deny_closing=deny_closing,
    )


def inference_events(events):
    return [
        event for event in events
        if event.get("eventType") == "operation.started"
        and (event.get("payload") or {}).get("operationKind") == "inference"
    ]


def completed_tool_events(events):
    return [
        event for event in events
        if event.get("eventType") == "operation.finished"
        and (event.get("payload") or {}).get("operationKind") == "tool_call"
        and (event.get("payload") or {}).get("status") == "succeeded"
        and (event.get("payload") or {}).get("resultSummaryKind") == "leaf_tool"
    ]


def classify(result):
    if result.kind == "invalid":
        return "invalid"
    if result.completion_source == "runtime_template":
        return "runtime_partial"
    if result.completion_kind == "partial":
        return "model_partial"
    return "full"


def expected_classification(item, enabled):
    return item["expectedCurrent" if enabled else "expectedControl"]


def first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return f"{path}:type"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}:keys"
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}:length"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = first_difference(
                left_item,
                right_item,
                f"{path}[{index}]",
            )
            if difference:
                return difference
        return None
    return None if left == right else path


def run_sample(service, backend_url, assistant_id, name, enabled):
    item = SCENARIOS[name]
    emitter = create_emitter(
        backend_url,
        assistant_id,
        item["denyClosing"],
    )
    token = activate_emitter(emitter)
    dispatched_requests = []
    actual_model_posts = 0

    original_post = llm_module._post

    def observe_post(url, payload, stream=False):
        nonlocal actual_model_posts
        if url.endswith("/v1/chat/completions") and not stream:
            dispatched_requests.append(copy.deepcopy(payload))
            actual_model_posts += 1
        return original_post(url, payload, stream=stream)

    def empty_post(url, payload, stream=False):
        if url.endswith("/v1/chat/completions") and not stream:
            dispatched_requests.append(copy.deepcopy(payload))
        return copy.deepcopy(EMPTY_RESPONSE)

    llm = TracingLlm(service, item["injectEmpty"], empty_post)
    calls = []
    raw_secret = "accessToken=mvp09-secret-must-not-leak"
    spec = AgentSpec(
        name="mvp09-benchmark",
        config_key="mvp09-benchmark",
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
        calls.append(index)
        if tool_name == "failing_probe":
            return {"error": "expected_probe_failure", "raw": raw_secret}
        return {
            "index": index,
            "validationMarker": item["marker"],
            "raw": raw_secret,
        }

    runtime_tools = {
        "collect_evidence": Tool(
            COLLECT_EVIDENCE,
            lambda *_args: {},
            lambda result: (
                f"Evidence {result['index']}: {result['validationMarker']}",
                None,
            ),
        ),
        "failing_probe": Tool(
            FAILING_PROBE,
            lambda *_args: {},
        ),
    }
    started = time.perf_counter()
    try:
        control_patch = (
            nullcontext()
            if enabled
            else patch("agents.loop._deterministic_partial", return_value=None)
        )
        with patch(
            "agents.loop.get_task_config",
            return_value={
                "max_rounds": item["maxRounds"],
                "normal_inference_soft_limit": 0,
                "max_tokens": 128,
                "max_tool_calls": item["maxToolCalls"],
                "tool_call_soft_limit": 0,
            },
        ), patch("agents.loop.get_llm_params", return_value={}), patch(
            "agents.loop.get_llm_service", return_value=llm
        ), patch.dict(REGISTRY, runtime_tools, clear=False), patch(
            "services.llm_service._post", side_effect=observe_post
        ), control_patch:
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
    events = request(
        backend_url,
        "GET",
        f"/executions/{emitter.context['rootExecutionId']}/events?limit=200",
    )["events"]
    artifacts = materialized_prompts(emitter.context["rootExecutionId"])
    completed = completed_tool_events(events)
    runtime_messages = [
        event for event in events
        if event.get("eventType") == "message.recorded"
        and (event.get("payload") or {}).get("generationSource")
        == "runtime_template"
    ]
    invalid_closings = [
        event for event in events
        if event.get("eventType") == "operation.finished"
        and (event.get("payload") or {}).get("operationKind") == "inference"
        and (event.get("payload") or {}).get("outcome") == "invalid"
    ]
    classification = classify(result)
    expected = expected_classification(item, enabled)
    expected_eligible = sum(
        1 for call in item["expectedCalls"]
        if name != "empty_closing_without_eligible_tool"
    )
    content = result.content or ""
    partial = result.partial_result or {}
    partial_operations = partial.get("completedOperations") or []
    partial_matches_events = (
        len(partial_operations) == len(completed)
        and all(
            operation.get("operationId") == event.get("operationId")
            and operation.get("toolCallId") == event.get("toolCallId")
            and operation.get("summary")
            == (event.get("payload") or {}).get("resultSummary")
            for operation, event in zip(partial_operations, completed)
        )
    )
    injected_closing_expected = item["injectEmpty"]
    runtime_expected = enabled and expected == "runtime_partial"
    marker_expected = expected in {"full", "model_partial", "runtime_partial"}
    correct = (
        classification == expected
        and calls == item["expectedCalls"]
        and llm.tool_batches == item["expectedBatches"]
        and len(completed) == expected_eligible
        and artifacts == dispatched_requests
        and raw_secret not in content
        and (item["marker"] in content) == marker_expected
        and bool(invalid_closings) == injected_closing_expected
        and len(runtime_messages) == int(runtime_expected)
    )
    if runtime_expected:
        expected_trigger = (
            "closing_unavailable"
            if item["denyClosing"]
            else "closing_output_empty"
        )
        correct = correct and (
            result.completion_kind == "partial"
            and result.completion_source == "runtime_template"
            and partial.get("trigger") == expected_trigger
            and partial_matches_events
            and partial.get("pending") == ["final_synthesis"]
            and "Pending:\n- Final synthesis" in content
        )
    if expected == "model_partial":
        correct = correct and (
            result.completion_source == "model"
            and result.completion_kind == "partial"
        )
    if item["denyClosing"]:
        correct = correct and (
            actual_model_posts == 1
            and len(inference_events(events)) == 1
            and emitter.preconsumed_closing_operation_id is not None
        )
    return {
        "mode": "mvp09" if enabled else "mvp08_control",
        "elapsedMs": elapsed_ms,
        "classification": classification,
        "completionKind": result.completion_kind,
        "completionReason": result.completion_reason,
        "completionSource": result.completion_source,
        "trigger": partial.get("trigger"),
        "methodCalls": llm.method_calls,
        "actualModelPosts": actual_model_posts,
        "inferenceStarts": len(inference_events(events)),
        "toolBatches": llm.tool_batches,
        "tools": calls,
        "eligibleTools": len(completed),
        "invalidClosings": len(invalid_closings),
        "runtimeMessages": len(runtime_messages),
        "partialMatchesEvents": partial_matches_events,
        "artifactsMatchDispatch": artifacts == dispatched_requests,
        "artifactDispatchDifference": first_difference(
            artifacts,
            dispatched_requests,
        ),
        "content": content,
        "correct": correct,
    }


def collect(service, backend_url, assistant_id, name, samples, max_attempts):
    return collect_paired(
        lambda enabled: run_sample(
            service, backend_url, assistant_id, name, enabled
        ),
        name,
        samples,
        max_attempts,
        ("mvp08_control", "mvp09"),
    )


def summarize(values, name, samples):
    common = latency_block(
        values,
        samples,
        control_mode="mvp08_control",
        current_mode="mvp09",
        control_key="mvp08Control",
        current_key="mvp09",
    )
    latency_applies = name in {
        "direct", "one_tool_then_answer", "valid_closing",
    }
    return {
        **common,
        "latencyThresholdApplies": latency_applies,
        "mvp09Classifications": accepted(
            values, "mvp09", "classification", samples
        ),
        "mvp09ModelPosts": accepted(
            values, "mvp09", "actualModelPosts", samples
        ),
        "mvp09InferenceStarts": accepted(
            values, "mvp09", "inferenceStarts", samples
        ),
        "mvp09EligibleTools": accepted(
            values, "mvp09", "eligibleTools", samples
        ),
        "mvp09Triggers": accepted(values, "mvp09", "trigger", samples),
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
    assistant = create_assistant(request, backend_url, "MVP 09 validation")
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
