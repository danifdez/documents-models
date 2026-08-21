import argparse
from pathlib import Path

from tests.execution.bench_harness import (
    accepted,
    create_assistant,
    deterministic_service,
    ensure_ingest_token,
    latency_block,
    resolve_backend_url,
    write_report,
)
from tests.execution import benchmark_exact_tool_repeat_block as base


def scenario(marker, instruction, **kwargs):
    return base.scenario(
        marker,
        instruction,
        control_block_enabled=True,
        terminate_enabled=True,
        **kwargs,
    )


SCENARIOS = {
    "direct": scenario(
        "MVP12-DIRECT-11",
        "Answer exactly MVP12-DIRECT-11 without using a tool.",
        tools=[], max_rounds=1, control_tools=[],
    ),
    "one_tool": scenario(
        "MVP12-ONE-23",
        "Call collect_evidence once with index 1. After its result answer "
        "exactly MVP12-ONE-23 without another tool.",
        tools=[base.COLLECT], max_rounds=2,
        control_tools=[["collect_evidence", 1]],
    ),
    "two_different_tools": scenario(
        "MVP12-TWO-31",
        "Call collect_evidence with index 1 and inspect_evidence with index 2 "
        "together. Then answer exactly MVP12-TWO-31.",
        tools=[base.COLLECT, base.INSPECT], max_rounds=2,
        control_tools=[["collect_evidence", 1], ["inspect_evidence", 2]],
    ),
    "repeat_recovers_after_warning": scenario(
        "MVP12-RECOVER-43",
        "The injector first calls collect_evidence twice with index 1. After "
        "the warning answer exactly MVP12-RECOVER-43 without another tool.",
        tools=[base.COLLECT], max_rounds=2,
        injected=[[('collect_evidence', 1), ('collect_evidence', 1)]],
        control_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        expected_guard=True, expected_warning=True,
        expected_current_outcome="normal_text",
        expected_control_outcome="normal_text",
    ),
    "block_then_final_text": scenario(
        "MVP12-BLOCK-FINAL-47",
        "The injector first repeats collect_evidence with index 1 through the "
        "first block. When you observe the blocked tool result, answer exactly "
        "MVP12-BLOCK-FINAL-47 without another tool.",
        tools=[base.COLLECT], max_rounds=3,
        injected=[
            [('collect_evidence', 1), ('collect_evidence', 1)],
            [('collect_evidence', 1)],
        ],
        control_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        expected_guard=True, expected_warning=True, expected_blocks=1,
        expected_current_outcome="normal_text",
        expected_control_outcome="normal_text",
    ),
    "block_then_different_tool": scenario(
        "MVP12-BLOCK-DIFFERENT-51",
        "The injector first repeats collect_evidence with index 1 through the "
        "first block. Then call inspect_evidence with index 2 once and answer "
        "exactly MVP12-BLOCK-DIFFERENT-51.",
        tools=[base.COLLECT, base.INSPECT], max_rounds=3,
        injected=[
            [('collect_evidence', 1), ('collect_evidence', 1)],
            [('collect_evidence', 1)], [('inspect_evidence', 2)],
        ],
        control_tools=[
            ["collect_evidence", 1], ["collect_evidence", 1],
            ["inspect_evidence", 2],
        ],
        expected_guard=True, expected_warning=True, expected_blocks=1,
        expected_partial=True,
    ),
    "same_batch_blocks": scenario(
        "MVP12-SAME-BATCH-55",
        "The injector repeats collect_evidence twice, then proposes two exact "
        "repeats in the same batch. After both blocked results answer exactly "
        "MVP12-SAME-BATCH-55.",
        tools=[base.COLLECT], max_rounds=3,
        injected=[
            [('collect_evidence', 1), ('collect_evidence', 1)],
            [('collect_evidence', 1), ('collect_evidence', 1)],
        ],
        control_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        expected_guard=True, expected_warning=True, expected_blocks=2,
        expected_current_outcome="normal_text",
        expected_control_outcome="normal_text",
    ),
    "mixed_batch": scenario(
        "MVP12-MIXED-59",
        "The injector repeats collect_evidence twice, then proposes its exact "
        "repeat with inspect_evidence index 2 in one batch. After the results "
        "answer exactly MVP12-MIXED-59.",
        tools=[base.COLLECT, base.INSPECT], max_rounds=3,
        injected=[
            [('collect_evidence', 1), ('collect_evidence', 1)],
            [('collect_evidence', 1), ('inspect_evidence', 2)],
        ],
        control_tools=[
            ["collect_evidence", 1], ["collect_evidence", 1],
            ["inspect_evidence", 2],
        ],
        expected_guard=True, expected_warning=True, expected_blocks=1,
        expected_current_outcome="normal_text",
        expected_control_outcome="normal_text",
    ),
    "block_applied_then_exact": scenario(
        "MVP12-TERMINATE-61",
        "The injector repeats collect_evidence through the first block. When "
        "you observe that blocked result, call collect_evidence once more with "
        "index 1 even though it was blocked.",
        tools=[base.COLLECT], max_rounds=3,
        injected=[
            [('collect_evidence', 1), ('collect_evidence', 1)],
            [('collect_evidence', 1)],
            {
                "calls": [('collect_evidence', 1)],
                "observeReal": True,
            },
        ],
        control_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        expected_guard=True, expected_warning=True,
        expected_blocks=1, expected_control_blocks=2,
        expected_terminations=1, expected_partial=True, parity=False,
        expected_current_outcome="loop_partial",
    ),
    "different_arguments": scenario(
        "MVP12-DIFFERENT-ARGS-67",
        "The injector repeats collect_evidence index 1 through a block, then "
        "calls collect_evidence with index 2. Answer exactly "
        "MVP12-DIFFERENT-ARGS-67 after that result.",
        tools=[base.COLLECT], max_rounds=3,
        injected=[
            [('collect_evidence', 1), ('collect_evidence', 1)],
            [('collect_evidence', 1)], [('collect_evidence', 2)],
        ],
        control_tools=[
            ["collect_evidence", 1], ["collect_evidence", 1],
            ["collect_evidence", 2],
        ],
        expected_guard=True, expected_warning=True, expected_blocks=1,
        expected_partial=True,
    ),
    "different_previous_result": scenario(
        "MVP12-DIFFERENT-RESULT-71",
        "The injector calls collect_evidence three times with index 1. Each "
        "result has a different revision. Then answer exactly "
        "MVP12-DIFFERENT-RESULT-71.",
        tools=[base.COLLECT], max_rounds=2,
        injected=[
            [
                ('collect_evidence', 1), ('collect_evidence', 1),
                ('collect_evidence', 1),
            ],
        ],
        control_tools=[["collect_evidence", 1]] * 3,
        result_mode="different",
        expected_guard=True, expected_warning=True,
    ),
    "no_eligible_summary": scenario(
        "MVP12-NO-SUMMARY-73",
        "The injector repeats collect_evidence through the first block and "
        "then proposes the exact call once more.",
        tools=[base.COLLECT], max_rounds=3,
        injected=[
            [('collect_evidence', 1), ('collect_evidence', 1)],
            [('collect_evidence', 1)], [('collect_evidence', 1)],
        ],
        control_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        result_mode="no_summary", expected_guard=True,
        expected_warning=True, expected_blocks=1, expected_control_blocks=2,
        expected_terminations=1, expected_partial=True, parity=False,
        expected_current_outcome="loop_failure",
    ),
    "tool_budget_precedes_termination": scenario(
        "MVP12-HARD-79",
        "The injector consumes one distinct tool call, then repeats "
        "collect_evidence through a block and proposes the exact call again. "
        "Use the available evidence to finish.",
        tools=[base.COLLECT, base.INSPECT], max_rounds=3,
        injected=[
            [
                ('inspect_evidence', 2), ('collect_evidence', 1),
                ('collect_evidence', 1),
            ],
            [('collect_evidence', 1)],
            [('collect_evidence', 1)],
        ],
        control_tools=[
            ["inspect_evidence", 2], ["collect_evidence", 1],
            ["collect_evidence", 1],
        ],
        max_tool_calls=3, expected_guard=True, expected_warning=True,
        expected_blocks=0, expected_hard_denials=1,
        expected_partial=True, parity=False,
    ),
    "terminal_policy_disabled": scenario(
        "MVP12-DISABLED-83",
        "The injector repeats collect_evidence through two blocked proposals. "
        "Then answer exactly MVP12-DISABLED-83.",
        tools=[base.COLLECT], max_rounds=3,
        injected=[
            [('collect_evidence', 1), ('collect_evidence', 1)],
            [('collect_evidence', 1)], [('collect_evidence', 1)],
        ],
        control_tools=[["collect_evidence", 1], ["collect_evidence", 1]],
        expected_guard=True, expected_warning=True, expected_blocks=2,
        expected_control_blocks=2, force_terminate_disabled=True,
        expected_partial=True,
    ),
}


def summarize(values, name, samples):
    item = SCENARIOS[name]
    common = latency_block(
        values,
        samples,
        control_mode="mvp11_control",
        current_mode="mvp12",
        control_key="mvp11Control",
        current_key="mvp12",
    )
    return {
        **common,
        "latencyThresholdApplies": item["parity"],
        "mvp12Tools": accepted(values, "mvp12", "tools", samples),
        "mvp12ProposedTools": accepted(
            values, "mvp12", "proposedTools", samples
        ),
        "mvp12Inferences": accepted(
            values, "mvp12", "inferences", samples
        ),
        "mvp12RealModelPosts": accepted(
            values, "mvp12", "realModelPosts", samples
        ),
        "mvp12ObservedRealResponses": accepted(
            values, "mvp12", "observedRealResponses", samples
        ),
        "mvp12BudgetUsage": accepted(
            values, "mvp12", "budgetUsage", samples
        ),
        "mvp12BlockSignals": accepted(
            values, "mvp12", "blockSignals", samples
        ),
        "mvp12TerminateSignals": accepted(
            values, "mvp12", "terminateSignals", samples
        ),
        "mvp12TerminateReservations": accepted(
            values, "mvp12", "terminateReservations", samples
        ),
        "mvp12BlockedLifecycleEvents": accepted(
            values, "mvp12", "blockedLifecycleEvents", samples
        ),
        "mvp12ResultKinds": accepted(
            values, "mvp12", "resultKind", samples
        ),
        "mvp12ResultReasons": accepted(
            values, "mvp12", "resultReason", samples
        ),
        "mvp12CompletionKinds": accepted(
            values, "mvp12", "completionKind", samples
        ),
        "mvp12CompletionReasons": accepted(
            values, "mvp12", "completionReason", samples
        ),
        "dispatchesAvoidedByTermination": sum(accepted(
            values, "mvp12", "terminateSignals", samples
        )),
        "passed": (
            not item["parity"]
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
    ensure_ingest_token()
    backend_url = resolve_backend_url()
    # Shares the MVP 11 client on purpose: both benchmarks report their
    # executions under the same backend workspace.
    assistant = create_assistant(base.request, backend_url, "MVP 12 validation")
    service, params = deterministic_service()
    selected = (
        {args.scenario: SCENARIOS[args.scenario]}
        if args.scenario else SCENARIOS
    )
    original = base.SCENARIOS
    base.SCENARIOS = SCENARIOS
    try:
        raw = {
            name: base.collect(
                service, backend_url, assistant["id"], name,
                args.samples, args.max_attempts,
                mode_names=("mvp11_control", "mvp12"),
            )
            for name in selected
        }
    finally:
        base.SCENARIOS = original
    scenarios = {
        name: summarize(values, name, args.samples)
        for name, values in raw.items()
    }
    false_terminals = sum(
        sum(accepted(values, "mvp12", "terminateSignals", args.samples))
        for name, values in raw.items()
        if SCENARIOS[name]["expectedTerminations"] == 0
    )
    report = {
        "model": Path(params["model_path"]).name,
        "samplesPerMode": args.samples,
        "modes": ["mvp11_control", "mvp12"],
        "scenarios": scenarios,
        "quality": {
            "terminatedDispatchesAvoided": sum(
                value["dispatchesAvoidedByTermination"]
                for value in scenarios.values()
            ),
            "falsePositiveTerminations": false_terminals,
        },
        "passed": (
            false_terminals == 0
            and all(value["passed"] for value in scenarios.values())
        ),
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
