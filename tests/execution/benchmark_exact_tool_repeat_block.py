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
    control_tools,
    mvp11_tools=None,
    injected=None,
    expected_guard=False,
    expected_warning=False,
    expected_blocks=0,
    expected_partial=False,
    parity=True,
    max_tool_calls=6,
    result_mode="stable",
    force_block_disabled=False,
    expected_hard_denials=0,
):
    return {
        "marker": marker,
        "system": f"/no_think\n{instruction}",
        "user": "Complete the validation sequence exactly as instructed.",
        "tools": tools,
        "maxRounds": max_rounds,
        "controlTools": control_tools,
        "mvp11Tools": mvp11_tools if mvp11_tools is not None else control_tools,
        "injected": injected or [],
        "expectedGuard": expected_guard,
        "expectedWarning": expected_warning,
        "expectedPartial": expected_partial,
        "expectedBlocks": expected_blocks,
        "parity": parity,
        "maxToolCalls": max_tool_calls,
        "resultMode": result_mode,
        "forceBlockDisabled": force_block_disabled,
        "expectedHardDenials": expected_hard_denials,
    }


SCENARIOS = {
    "direct": scenario(
        "MVP11-DIRECT-11",
        "Answer exactly MVP11-DIRECT-11 without using a tool.",
        tools=[],
        max_rounds=1,
        control_tools=[],
    ),
    "one_tool": scenario(
        "MVP11-ONE-23",
        "Call collect_evidence once with index 1. After its result answer "
        "exactly MVP11-ONE-23 without another tool.",
        tools=[COLLECT],
        max_rounds=2,
        control_tools=[["collect_evidence", 1]],
    ),
    "two_different_tools": scenario(
        "MVP11-TWO-31",
        "Call collect_evidence with index 1 and inspect_evidence with index 2 "
        "together in the first response. Then answer exactly MVP11-TWO-31.",
        tools=[COLLECT, INSPECT],
        max_rounds=2,
        control_tools=[["collect_evidence", 1], ["inspect_evidence", 2]],
    ),
    "repeat_recovers_after_warning": scenario(
        "MVP11-RECOVER-43",
        "A test injector will first call collect_evidence twice with index 1. "
        "After those results, if a system message says an exact tool call was "
        "repeated, answer exactly MVP11-RECOVER-43. Without that warning call "
        "collect_evidence once more with index 1, then answer the marker.",
        tools=[COLLECT],
        max_rounds=4,
        injected=[[('collect_evidence', 1)], [('collect_evidence', 1)]],
        control_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        expected_guard=True,
        expected_warning=True,
    ),
    "repeat_persists_once": scenario(
        "MVP11-BLOCK-47",
        "A test injector will first call collect_evidence twice with index 1. "
        "After those results call collect_evidence once more with index 1 even "
        "if warned. If that call is blocked, answer exactly MVP11-BLOCK-47.",
        tools=[COLLECT],
        max_rounds=4,
        injected=[[('collect_evidence', 1)], [('collect_evidence', 1)]],
        control_tools=[
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["collect_evidence", 1],
        ],
        mvp11_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        expected_guard=True,
        expected_warning=True,
        expected_blocks=1,
        expected_partial=True,
        parity=False,
    ),
    "two_persistent_proposals": scenario(
        "MVP11-TWO-BLOCKS-51",
        "A test injector will first call collect_evidence twice with index 1. "
        "After those results call collect_evidence twice with index 1 together "
        "in one response even if warned. If blocked, answer exactly "
        "MVP11-TWO-BLOCKS-51.",
        tools=[COLLECT],
        max_rounds=4,
        injected=[
            [('collect_evidence', 1)],
            [('collect_evidence', 1)],
            [('collect_evidence', 1), ('collect_evidence', 1)],
        ],
        control_tools=[
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["collect_evidence", 1],
        ],
        mvp11_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        expected_guard=True,
        expected_warning=True,
        expected_blocks=2,
        expected_partial=True,
        parity=False,
    ),
    "persistent_and_distinct_batch": scenario(
        "MVP11-MIXED-55",
        "A test injector will first call collect_evidence twice with index 1. "
        "Then call collect_evidence with index 1 and inspect_evidence with "
        "index 2 together. After the results answer exactly MVP11-MIXED-55.",
        tools=[COLLECT, INSPECT],
        max_rounds=4,
        injected=[
            [('collect_evidence', 1)],
            [('collect_evidence', 1)],
            [('collect_evidence', 1), ('inspect_evidence', 2)],
        ],
        control_tools=[
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["inspect_evidence", 2],
        ],
        mvp11_tools=[
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["inspect_evidence", 2],
        ],
        expected_guard=True,
        expected_warning=True,
        expected_blocks=1,
        expected_partial=True,
        parity=False,
    ),
    "change_arguments_after_warning": scenario(
        "MVP11-CHANGE-59",
        "A test injector will first call collect_evidence twice with index 1. "
        "After the warning call collect_evidence with index 2, then answer "
        "exactly MVP11-CHANGE-59.",
        tools=[COLLECT],
        max_rounds=4,
        injected=[[('collect_evidence', 1)], [('collect_evidence', 1)]],
        control_tools=[
            ["collect_evidence", 1],
            ["collect_evidence", 1],
            ["collect_evidence", 2],
        ],
        expected_guard=True,
        expected_warning=True,
        expected_partial=True,
    ),
    "same_input_different_results": scenario(
        "MVP11-DIFFERENT-RESULT-63",
        "A test injector will first call collect_evidence twice with index 1. "
        "The observations differ. After the warning call it once more with "
        "index 1, then answer exactly MVP11-DIFFERENT-RESULT-63.",
        tools=[COLLECT],
        max_rounds=4,
        injected=[[('collect_evidence', 1)], [('collect_evidence', 1)]],
        control_tools=[["collect_evidence", 1]] * 3,
        expected_guard=True,
        expected_warning=True,
        expected_partial=True,
        result_mode="different",
    ),
    "polling_state_changes": scenario(
        "MVP11-POLLING-67",
        "A test injector will first call collect_evidence twice with index 1. "
        "Each observation is a new state. After the warning call it once more "
        "with index 1, then answer exactly MVP11-POLLING-67.",
        tools=[COLLECT],
        max_rounds=4,
        injected=[[('collect_evidence', 1)], [('collect_evidence', 1)]],
        control_tools=[["collect_evidence", 1]] * 3,
        expected_guard=True,
        expected_warning=True,
        expected_partial=True,
        result_mode="polling",
    ),
    "failed_tool_repeated": scenario(
        "MVP11-FAILED-71",
        "Call failing_probe with index 1. Its error is expected; repeat exactly "
        "the same failing_probe once. After the second error answer exactly "
        "MVP11-FAILED-71.",
        tools=[FAIL],
        max_rounds=3,
        control_tools=[["failing_probe", 1], ["failing_probe", 1]],
    ),
    "hard_budget_precedes_block": scenario(
        "MVP11-HARD-73",
        "A test injector will call collect_evidence twice with index 1. After "
        "the warning request it once more. The runtime must use protected "
        "finalization and answer with the available evidence.",
        tools=[COLLECT],
        max_rounds=4,
        control_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        injected=[
            [('collect_evidence', 1)],
            [('collect_evidence', 1)],
            [('collect_evidence', 1)],
        ],
        expected_guard=True,
        expected_warning=True,
        expected_partial=True,
        parity=False,
        max_tool_calls=2,
        expected_hard_denials=1,
    ),
    "block_then_protected_close": scenario(
        "MVP11-CLOSE-79",
        "A test injector will call collect_evidence twice with index 1. After "
        "the warning request it once more. If blocked, use protected "
        "finalization with the available evidence.",
        tools=[COLLECT],
        max_rounds=3,
        control_tools=[["collect_evidence", 1]] * 3,
        mvp11_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        injected=[
            [('collect_evidence', 1)],
            [('collect_evidence', 1)],
            [('collect_evidence', 1)],
        ],
        expected_guard=True,
        expected_warning=True,
        expected_blocks=1,
        expected_partial=True,
        parity=False,
        max_tool_calls=3,
    ),
    "policy_disabled": scenario(
        "MVP11-DISABLED-83",
        "A test injector will call collect_evidence twice with index 1. After "
        "the warning call it once more with index 1, then answer exactly "
        "MVP11-DISABLED-83.",
        tools=[COLLECT],
        max_rounds=4,
        control_tools=[["collect_evidence", 1]] * 3,
        injected=[
            [('collect_evidence', 1)],
            [('collect_evidence', 1)],
            [('collect_evidence', 1)],
        ],
        expected_guard=True,
        expected_warning=True,
        expected_partial=True,
        force_block_disabled=True,
    ),
}


def openai_tool_response(calls, inference):
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": f"mvp11-injected-{inference}-{position}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps({"index": tool_index}),
                    },
                } for position, (name, tool_index) in enumerate(calls, 1)],
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
        self.proposed_tools = []
        self.technical_results = {}

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
        for item in messages:
            if item.get("role") != "tool":
                continue
            try:
                payload = json.loads(item.get("content") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("error") == "immediate_exact_tool_repeat_blocked":
                self.technical_results[str(item.get("tool_call_id") or "")] = payload

    def chat_with_tools(self, messages, *args, **kwargs):
        self.inferences += 1
        self._observe(messages)
        if self.injected:
            response = openai_tool_response(
                self.injected.pop(0), self.inferences,
            )
            with patch("services.llm_service._post", return_value=response):
                value = self.delegate.chat_with_tools(messages, *args, **kwargs)
        else:
            self.real_model_posts += 1
            value = self.delegate.chat_with_tools(messages, *args, **kwargs)
        calls = value.get("tool_calls") or []
        for call in calls:
            fn = call.get("function") or {}
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            self.proposed_tools.append([
                str(fn.get("name") or ""),
                int(arguments.get("index") or 0),
            ])
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
            "X-Workspace-Id": "mvp11-benchmark",
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
        {"content": "MVP 11 exact tool repeat block benchmark probe"},
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


def block_enabled(item, enabled):
    return enabled and not item["forceBlockDisabled"]


def expected_calls(item, enabled):
    return item["mvp11Tools"] if block_enabled(item, enabled) else item["controlTools"]


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
        name="mvp11-benchmark",
        config_key="mvp11-benchmark",
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
        result = {
            "index": index,
            "validationMarker": item["marker"],
        }
        if item["resultMode"] == "different":
            result["revision"] = len(calls)
        elif item["resultMode"] == "polling":
            result["state"] = f"poll-{len(calls)}"
        return result

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
                "exact_tool_repeat_warning": True,
                "exact_tool_repeat_block_after_warning": block_enabled(
                    item, enabled
                ),
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
    warning_signals = [signal for signal in signals if signal.get("action") == "warn"]
    block_signals = [signal for signal in signals if signal.get("action") == "block"]
    starts = warning_starts(events)
    prompt_warning_count = sum(
        str(message.get("content") or "").startswith(REPEAT_WARNING)
        for prompt in prompts
        for message in prompt.get("messages", [])
        if isinstance(message, dict)
    )
    expected_signal_count = 1 if item["expectedGuard"] else 0
    expected_warning_count = 1 if item["expectedWarning"] else 0
    expected_block_count = (
        item["expectedBlocks"] if block_enabled(item, enabled) else 0
    )
    content = result.content or ""
    partial_matches = (
        result.kind == "final_text"
        and result.completion_kind == "partial"
        and bool(content.strip())
        if item["expectedPartial"]
        else result.kind == "final_text"
        and result.completion_kind is None
        and item["marker"] in content
    )
    actual_calls = normalized_calls(calls)
    calls_match = actual_calls == expected_calls(item, enabled)
    guard_state = ((progress.get("ledger") or {}).get("loopGuards") or {})
    guard = next(iter(guard_state.values()), {}).get("exactToolRepeat", {})
    warning_pending_expected = bool(item["expectedGuard"] and not item["expectedWarning"])
    operation_budget = ((progress.get("ledger") or {}).get("operationBudget") or {})
    reservations = list((operation_budget.get("reservations") or {}).values())
    block_reservations = [
        reservation for reservation in reservations
        if reservation.get("reason") == "immediate_exact_tool_repeat_blocked"
    ]
    hard_denials = [
        reservation for reservation in reservations
        if reservation.get("reason") == "tool_budget_hard_limit_reached"
    ]
    blocked_ids = {
        signal.get("triggeringOperationId") for signal in block_signals
    }
    blocked_lifecycle_events = [
        event for event in events
        if event.get("operationId") in blocked_ids
        and event.get("eventType") in {
            "operation.started", "operation.finished", "source.observed",
        }
    ]
    prompt_technical = {}
    for prompt in prompts:
        for message in prompt.get("messages", []):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            try:
                payload = json.loads(message.get("content") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("error") == "immediate_exact_tool_repeat_blocked":
                prompt_technical[str(message.get("tool_call_id") or "")] = payload
    technical_safe = all(
        set(payload) == {"error", "blocked", "message"}
        and payload.get("blocked") is True
        for payload in [*prompt_technical.values(), *llm.technical_results.values()]
    )
    grant = next(iter((operation_budget.get("grants") or {}).values()), {})
    effective_block = bool(
        (grant.get("effectivePolicy") or {}).get(
            "exactToolRepeatBlockAfterWarning", False
        )
    )
    proposed_expected = (
        len(expected_calls(item, enabled))
        + expected_block_count
        + item["expectedHardDenials"]
    )
    usage = budget_usage(progress)
    correct = (
        partial_matches
        and calls_match
        and len(warning_signals) == expected_signal_count
        and len(block_signals) == expected_block_count
        and len(starts) == expected_warning_count
        and prompt_warning_count == expected_warning_count
        and bool(guard.get("warningPending", False)) == warning_pending_expected
        and int(guard.get("blocks", 0)) == expected_block_count
        and len(block_reservations) == expected_block_count
        and len(hard_denials) == item["expectedHardDenials"]
        and not blocked_lifecycle_events
        and len(prompt_technical) == expected_block_count
        and len(llm.technical_results) == expected_block_count
        and technical_safe
        and effective_block == block_enabled(item, enabled)
        and len(llm.proposed_tools) == proposed_expected
        and usage is not None
        and usage["tool"] == len(actual_calls)
    )
    return {
        "mode": "mvp11" if enabled else "mvp10_control",
        "elapsedMs": elapsed_ms,
        "realModelPosts": llm.real_model_posts,
        "inferences": llm.inferences,
        "toolBatches": llm.tool_batches,
        "tools": actual_calls,
        "expectedTools": expected_calls(item, enabled),
        "proposedTools": llm.proposed_tools,
        "resultKind": result.kind,
        "completionKind": result.completion_kind,
        "completionReason": result.completion_reason,
        "content": content,
        "warningSignals": len(warning_signals),
        "blockSignals": len(block_signals),
        "blockReservations": len(block_reservations),
        "hardLimitDenials": len(hard_denials),
        "blockedLifecycleEvents": len(blocked_lifecycle_events),
        "technicalResults": len(llm.technical_results),
        "technicalPromptResults": len(prompt_technical),
        "technicalSafe": technical_safe,
        "warningStarts": len(starts),
        "warningPrompts": prompt_warning_count,
        "warningInferences": llm.warning_inferences,
        "warningPending": bool(guard.get("warningPending", False)),
        "budgetUsage": usage,
        "rootExecutionId": context["rootExecutionId"],
        "correct": correct,
    }


def collect(service, backend_url, assistant_id, name, samples, max_attempts):
    values = []
    accepted = {"mvp10_control": 0, "mvp11": 0}
    for attempt in range(max_attempts):
        order = (False, True) if attempt % 2 == 0 else (True, False)
        for enabled in order:
            mode = "mvp11" if enabled else "mvp10_control"
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
    control = accepted(values, "mvp10_control", "elapsedMs", samples)
    current = accepted(values, "mvp11", "elapsedMs", samples)
    control_median = statistics.median(control)
    current_median = statistics.median(current)
    delta = current_median - control_median
    threshold = max(round(control_median * 0.10), 150)
    parity = item["parity"]
    return {
        "mvp10ControlMs": control,
        "mvp11Ms": current,
        "mvp10ControlMedianMs": control_median,
        "mvp11MedianMs": current_median,
        "deltaMs": delta,
        "thresholdMs": threshold,
        "latencyThresholdApplies": parity,
        "mvp11Tools": accepted(values, "mvp11", "tools", samples),
        "mvp11ProposedTools": accepted(
            values, "mvp11", "proposedTools", samples
        ),
        "mvp11Inferences": accepted(values, "mvp11", "inferences", samples),
        "mvp11RealModelPosts": accepted(
            values, "mvp11", "realModelPosts", samples
        ),
        "mvp11BudgetUsage": accepted(values, "mvp11", "budgetUsage", samples),
        "mvp11WarningSignals": accepted(
            values, "mvp11", "warningSignals", samples
        ),
        "mvp11BlockSignals": accepted(
            values, "mvp11", "blockSignals", samples
        ),
        "mvp11TechnicalResults": accepted(
            values, "mvp11", "technicalResults", samples
        ),
        "dispatchesAvoided": samples * (
            len(item["controlTools"]) - len(item["mvp11Tools"])
        ) if not item["forceBlockDisabled"] else 0,
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
        {"name": "MVP 11 validation", "systemPrompt": "Validation fixture"},
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
    blocked = sum(
        value["dispatchesAvoided"] for value in scenarios.values()
    )
    false_positive_blocks = sum(
        value["blockSignals"]
        for name, values in raw.items()
        for value in values
        if value["mode"] == "mvp11"
        and SCENARIOS[name]["expectedBlocks"] == 0
    )
    report = {
        "model": Path(params["model_path"]).name,
        "samplesPerMode": args.samples,
        "scenarios": scenarios,
        "quality": {
            "blockedDispatches": blocked,
            "falsePositiveBlocks": false_positive_blocks,
        },
        "passed": (
            false_positive_blocks == 0
            and all(value["passed"] for value in scenarios.values())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
