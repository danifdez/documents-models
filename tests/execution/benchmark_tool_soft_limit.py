import argparse
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

from agents.loop import run_agent_loop
from lib.execution import ExecutionEmitter, activate_emitter, reset_emitter
from lib.framework.agent import AgentSpec
from lib.framework.tool import ToolContext
from lib.llm.config import get_llm_params
from services.llm_service import get_llm_service
from tests.execution.benchmark_inference_budget import activate_execution
from tests.execution.benchmark_tool_budget import MeasuredEmitter, request


WARNING_PREFIX = "Tool budget is low: 2 of 6 calls remain."
TOOL = {
    "type": "function",
    "function": {
        "name": "collect_evidence",
        "description": (
            "Collect one required validation evidence item by numeric index. "
            "Independent indices should be requested together in one response."
        ),
        "parameters": {
            "type": "object",
            "properties": {"index": {"type": "integer", "minimum": 1}},
            "required": ["index"],
        },
    },
}
INITIAL_TOOL = {
    "type": "function",
    "function": {
        "name": "collect_initial",
        "description": "Collect one of the four independent initial evidence items.",
        "parameters": {
            "type": "object",
            "properties": {"index": {"type": "integer", "minimum": 1}},
            "required": ["index"],
        },
    },
}
FOLLOWUP_TOOL = {
    "type": "function",
    "function": {
        "name": "collect_followup",
        "description": (
            "Collect follow-up evidence. This requires the authorization code "
            "returned by collect_initial index 4."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 5},
                "authorizationCode": {"type": "string"},
            },
            "required": ["index", "authorizationCode"],
        },
    },
}


def scenario(marker, instruction, batches, tools, inferences, usage, partial=False):
    return {
        "marker": marker,
        "system": f"/no_think\n{instruction}",
        "user": "Complete the validation sequence exactly as instructed.",
        "batches": batches,
        "tools": tools,
        "inferences": inferences,
        "usage": usage,
        "partial": partial,
    }


SCENARIOS = {
    "direct": scenario(
        "MVP07-DIRECT-11",
        "Answer exactly MVP07-DIRECT-11. Do not call collect_evidence.",
        [0], [], 1,
        {"normal": 1, "repair": 0, "closing": 0, "tool": 0},
    ),
    "one_tool": scenario(
        "MVP07-ONE-23",
        "Call collect_evidence once with index 1. After its result, answer "
        "exactly MVP07-ONE-23.",
        [1, 0], [1], 2,
        {"normal": 2, "repair": 0, "closing": 0, "tool": 1},
    ),
    "three_tools_below_soft_limit": scenario(
        "MVP07-THREE-31",
        "In the first response call collect_evidence three times together, with "
        "indices 1, 2, and 3. After all results, answer exactly MVP07-THREE-31.",
        [3, 0], [1, 2, 3], 2,
        {"normal": 2, "repair": 0, "closing": 0, "tool": 3},
    ),
    "four_tools_then_close": scenario(
        "MVP07-FOUR-43",
        "In the first response call collect_evidence four times together, with "
        "indices 1, 2, 3, and 4. After all results, answer exactly MVP07-FOUR-43 "
        "without another tool.",
        [4, 0], [1, 2, 3, 4], 2,
        {"normal": 2, "repair": 0, "closing": 0, "tool": 4},
    ),
    "fifth_tool_is_essential": scenario(
        "MVP07-FIVE-53",
        "In the first response call collect_initial four times together, with "
        "indices 1, 2, 3, and 4. Do not call collect_followup yet: it requires "
        "the authorization code returned by index 4. After those results, index "
        "5 remains essential: call collect_followup once with index 5 and that "
        "authorization code. Then answer exactly MVP07-FIVE-53.",
        [4, 1, 0], [1, 2, 3, 4, 5], 3,
        {"normal": 3, "repair": 0, "closing": 0, "tool": 5},
    ),
    "hard_limit_after_warning": scenario(
        "MVP07-HARD-67",
        "In the first response call collect_initial four times together, with "
        "indices 1, 2, 3, and 4. Do not call collect_followup yet: it requires "
        "the authorization code returned by index 4. After those results call "
        "collect_followup three times together with indices 5, 6, and 7 and that "
        "authorization code. If the tool budget stops a call and requires a final "
        "answer without tools, answer exactly MVP07-HARD-67.",
        [4, 3], [1, 2, 3, 4, 5, 6], 3,
        {"normal": 2, "repair": 0, "closing": 1, "tool": 6},
        partial=True,
    ),
}


class TracingLlm:
    def __init__(self, delegate):
        self.delegate = delegate
        self.inferences = 0
        self.tool_batches = []
        self.warning_inferences = 0
        self.warning_messages = []

    def _observe(self, messages):
        warnings = [
            item.get("content", "")
            for item in messages
            if item.get("role") == "system"
            and item.get("content", "").startswith("Tool budget is low:")
        ]
        if warnings:
            self.warning_inferences += 1
            self.warning_messages.extend(warnings)

    def chat_with_tools(self, messages, *args, **kwargs):
        self.inferences += 1
        self._observe(messages)
        response = self.delegate.chat_with_tools(messages, *args, **kwargs)
        self.tool_batches.append(len(response.get("tool_calls") or []))
        return response

    def chat(self, messages, *args, **kwargs):
        self.inferences += 1
        self._observe(messages)
        return self.delegate.chat(messages, *args, **kwargs)


def create_emitter(backend_url, assistant_id):
    created = request(
        backend_url,
        "POST",
        f"/assistants/{assistant_id}/messages",
        {"content": "MVP 07 tool soft limit benchmark probe"},
    )
    return MeasuredEmitter(activate_execution(created["executionId"]))


def budget_state(progress):
    grants = (((progress.get("ledger") or {}).get("operationBudget") or {})
              .get("grants", {}))
    if len(grants) != 1:
        return None, None
    grant = next(iter(grants.values()))
    usage = grant.get("usage") or {}
    consumed = {
        bucket: int((usage.get(bucket) or {}).get("consumed", 0))
        for bucket in ("normal", "repair", "closing", "tool")
    }
    return consumed, usage.get("tool") or {}


def event_count(events, kind):
    return sum(
        event.get("eventType") == "progress.reported"
        and (event.get("payload") or {}).get("kind") == kind
        for event in events
    )


def run_sample(service, backend_url, assistant_id, name, soft_enabled):
    item = SCENARIOS[name]
    if name == "direct":
        offered_tools = []
    elif name in {"fifth_tool_is_essential", "hard_limit_after_warning"}:
        offered_tools = [INITIAL_TOOL, FOLLOWUP_TOOL]
    else:
        offered_tools = [TOOL]
    emitter = create_emitter(backend_url, assistant_id)
    token = activate_emitter(emitter)
    llm = TracingLlm(service)
    calls = []
    call_names = []
    spec = AgentSpec(
        name="mvp07-benchmark",
        config_key="mvp07-benchmark",
        system_prompt=item["system"],
        tool_names=frozenset(
            tool["function"]["name"] for tool in offered_tools
        ),
        max_rounds=4,
        fallback_max_tokens=128,
    )
    messages = [
        {"role": "system", "content": item["system"]},
        {"role": "user", "content": item["user"]},
    ]

    def dispatch(_name, arguments, _ctx):
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        index = int(arguments.get("index") or 0)
        calls.append(index)
        call_names.append(_name)
        result = {"index": index, "validationMarker": item["marker"]}
        if _name == "collect_initial" and index == 4:
            result["authorizationCode"] = "MVP07-FOLLOWUP-AUTH"
        return result

    started = time.perf_counter()
    try:
        with patch(
            "agents.loop.get_task_config",
            return_value={
                "max_rounds": 4,
                "max_tokens": 128,
                "max_tool_calls": 6,
                "tool_call_soft_limit": 4 if soft_enabled else 0,
            },
        ), patch("agents.loop.get_llm_params", return_value={}), patch(
            "agents.loop.get_llm_service", return_value=llm
        ):
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
    usage, tool_state = budget_state(progress)
    signal_count = event_count(events, "budget_soft_limit_reached")
    reached = len(item["tools"]) >= 4
    expected_warning_count = 1 if soft_enabled and reached else 0
    expected_signal_count = 1 if soft_enabled and reached else 0
    expected_partial = item["partial"]
    expected_call_names = (
        ["collect_initial"] * 4
        + ["collect_followup"] * (len(item["tools"]) - 4)
        if name in {"fifth_tool_is_essential", "hard_limit_after_warning"}
        else ["collect_evidence"] * len(item["tools"])
    )
    classification_matches = (
        result.completion_kind == "partial"
        and result.completion_reason == "tool_budget_exhausted"
        if expected_partial
        else result.completion_kind is None and result.completion_reason is None
    )
    correct = (
        result.kind == "final_text"
        and item["marker"] in (result.content or "")
        and calls == item["tools"]
        and call_names == expected_call_names
        and llm.tool_batches == item["batches"]
        and llm.inferences == item["inferences"]
        and usage == item["usage"]
        and signal_count == expected_signal_count
        and llm.warning_inferences == expected_warning_count
        and all(message.startswith(WARNING_PREFIX)
                for message in llm.warning_messages)
        and bool(tool_state.get("softLimitReached"))
        == bool(soft_enabled and reached)
        and classification_matches
    )
    return {
        "mode": "mvp07" if soft_enabled else "mvp06_control",
        "elapsedMs": elapsed_ms,
        "toolReservationMs": emitter.tool_reservation_ms,
        "inferences": llm.inferences,
        "toolBatches": llm.tool_batches,
        "tools": calls,
        "toolNames": call_names,
        "content": result.content or "",
        "completionKind": result.completion_kind,
        "completionReason": result.completion_reason,
        "budgetUsage": usage,
        "toolBudgetState": tool_state,
        "softLimitSignals": signal_count,
        "warningInferences": llm.warning_inferences,
        "warningMessages": llm.warning_messages,
        "correct": correct,
    }


def collect(service, backend_url, assistant_id, name, samples, max_attempts):
    values = []
    accepted = {"mvp06_control": 0, "mvp07": 0}
    for attempt in range(max_attempts):
        order = (False, True) if attempt % 2 == 0 else (True, False)
        for soft_enabled in order:
            mode = "mvp07" if soft_enabled else "mvp06_control"
            if accepted[mode] >= samples:
                continue
            sample = run_sample(
                service, backend_url, assistant_id, name, soft_enabled
            )
            values.append(sample)
            print(name, json.dumps(sample, ensure_ascii=False), flush=True)
            if sample["correct"]:
                accepted[mode] += 1
        if all(value >= samples for value in accepted.values()):
            break
    if any(value < samples for value in accepted.values()):
        raise RuntimeError(f"{name}: insufficient correct samples: {accepted}")
    return values


def accepted(values, mode, key, samples):
    return [
        value[key]
        for value in values
        if value["correct"] and value["mode"] == mode
    ][:samples]


def summarize(values, name, samples):
    control = accepted(values, "mvp06_control", "elapsedMs", samples)
    current = accepted(values, "mvp07", "elapsedMs", samples)
    control_median = statistics.median(control)
    current_median = statistics.median(current)
    delta = current_median - control_median
    threshold = max(round(control_median * 0.10), 150)
    latency_applies = name in {
        "direct", "one_tool", "three_tools_below_soft_limit",
    }
    passed = not latency_applies or delta <= threshold
    return {
        "mvp06ControlMs": control,
        "mvp07Ms": current,
        "mvp06ControlMedianMs": control_median,
        "mvp07MedianMs": current_median,
        "deltaMs": delta,
        "thresholdMs": threshold,
        "latencyThresholdApplies": latency_applies,
        "mvp07ToolReservationMs": accepted(
            values, "mvp07", "toolReservationMs", samples
        ),
        "mvp07Inferences": accepted(values, "mvp07", "inferences", samples),
        "mvp07Tools": accepted(values, "mvp07", "tools", samples),
        "mvp07Warnings": accepted(
            values, "mvp07", "warningInferences", samples
        ),
        "mvp07Signals": accepted(
            values, "mvp07", "softLimitSignals", samples
        ),
        "mvp07CompletionKinds": accepted(
            values, "mvp07", "completionKind", samples
        ),
        "mvp07CompletionReasons": accepted(
            values, "mvp07", "completionReason", samples
        ),
        "mvp07BudgetUsage": accepted(
            values, "mvp07", "budgetUsage", samples
        ),
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend_url = os.environ.get(
        "BACKEND_URL", "http://127.0.0.1:3000"
    ).rstrip("/")
    assistant = request(
        backend_url,
        "POST",
        "/assistants",
        {"name": "MVP 07 validation", "systemPrompt": "Validation fixture"},
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
        name: collect(
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
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
