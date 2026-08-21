import json
import os
import subprocess
import sys
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from config import EXECUTIONS_TABLE
from lib.execution import ExecutionEmitter
from lib.execution.emitter import canonical_tool_input_fingerprint
from tests.execution import test_exact_tool_repeat_block_backend_e2e as block_e2e


_CHILD_CODE = r'''
import json
import os

from lib.execution import ExecutionEmitter, ToolLoopGuardBlocked
from lib.execution.progress import ProgressLoopContext

context = json.loads(os.environ["MVP12_EXECUTION_CONTEXT"])
phase = os.environ["MVP12_INTERRUPT_PHASE"]
emitter = ExecutionEmitter(context)

if phase == "after_block":
    progress = ProgressLoopContext.start(
        emitter,
        agent_name="mvp12-interruption",
        loop_kind="top_level",
        max_rounds=3,
        normal_inference_soft_limit=0,
        max_output_repairs=0,
        forced_finalization_available=True,
        max_tokens_per_inference=64,
        max_tool_calls=4,
        tool_call_soft_limit=0,
        exact_tool_repeat_warning=True,
        exact_tool_repeat_block_after_warning=True,
        exact_tool_repeat_terminate_after_block=True,
    )
    arguments = {"index": 1, "nested": {"alpha": "β"}}
    for batch_index in (0, 1):
        handle = emitter.start_tool(
            "collect_evidence",
            arguments,
            f"mvp12-provider-{batch_index}",
            progress.trace(round=1, phase="agent_loop"),
            repeat_comparable=True,
            tool_batch_size=2,
            tool_batch_index=batch_index,
        )
        emitter.flush_evidence()
        result = {"validationMarker": "MVP12-CUT-DURABLE"}
        source = emitter.observe_tool_result(handle, result)
        emitter.finish_tool(
            handle,
            result,
            source,
            result_summary="Evidence collected: MVP12-CUT-DURABLE",
            result_summary_kind="leaf_tool",
        )
        emitter.flush_evidence()
    request = {"messages": [{"role": "user", "content": "continue"}]}
    inference = emitter.start_inference(
        "chat_with_tools",
        request,
        progress.trace(round=2, phase="agent_loop"),
    )
    emitter.flush_evidence()
    emitter.finish_inference(
        inference,
        {"content": "repeat once"},
        outcome="tool_requests",
    )
    emitter.flush_evidence()
    try:
        emitter.start_tool(
            "collect_evidence",
            arguments,
            "mvp12-blocked-call",
            progress.trace(round=2, phase="agent_loop"),
            repeat_comparable=True,
            tool_batch_size=1,
            tool_batch_index=0,
        )
    except ToolLoopGuardBlocked:
        os._exit(91)
    raise AssertionError("the exact repeat was not blocked")

grant_id = os.environ["MVP12_GRANT_ID"]
trace = {
    "loopId": context["executionId"],
    "agentName": "mvp12-interruption",
    "loopKind": "top_level",
    "maxRounds": 3,
    "round": 3,
    "phase": "agent_loop",
    "budgetGrantId": grant_id,
}
if phase == "after_application":
    request = {"messages": [{"role": "tool", "content": "blocked"}]}
    inference = emitter.start_inference("chat_with_tools", request, trace)
    emitter.flush_evidence()
    emitter.finish_inference(
        inference,
        {"content": "repeat again"},
        outcome="tool_requests",
    )
    emitter.flush_evidence()
    os._exit(91)

body = json.loads(os.environ["MVP12_TERMINAL_BODY"])
decision = emitter._post("progress/reservations", body)
assert decision["granted"] is False
assert decision["reservation"]["reason"] == (
    "immediate_exact_tool_repeat_terminated"
)
os._exit(91)
'''


@unittest.skipUnless(
    os.environ.get("RUN_EXACT_TOOL_REPEAT_TERMINATION_BACKEND_E2E") == "1",
    "set RUN_EXACT_TOOL_REPEAT_TERMINATION_BACKEND_E2E=1 with the backend running",
)
class ExactToolRepeatTerminationInterruptionTests(unittest.TestCase):
    backend_url = os.environ.get(
        "BACKEND_URL", "http://127.0.0.1:3000"
    ).rstrip("/")

    @classmethod
    def setUpClass(cls):
        block_e2e.ensure_ingest_token()

    def request(self, method, path, body=None):
        return block_e2e.ExactToolRepeatBlockInterruptionTests.request(
            self, method, path, body
        )

    def database_connection(self):
        return block_e2e.ExactToolRepeatBlockInterruptionTests.database_connection(
            self
        )

    def create_execution(self):
        assistant = self.request(
            "POST",
            "/assistants",
            {"name": "MVP 12 interruption", "systemPrompt": "Validation fixture"},
        )
        execution = self.request(
            "POST",
            f"/assistants/{assistant['id']}/messages",
            {"content": "MVP 12 interruption probe"},
        )
        attempt_id = str(uuid.uuid4())
        with self.database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {EXECUTIONS_TABLE}
                SET status = 'running', phase = 'worker_execution',
                    attempt_id = %s, started_at = COALESCE(started_at, now()),
                    updated_at = now()
                WHERE execution_id = %s
                RETURNING root_execution_id, execution_id, turn_id, last_event_id
                """,
                (attempt_id, execution["executionId"]),
            )
            row = cursor.fetchone()
        return {
            "rootExecutionId": str(row[0]),
            "executionId": str(row[1]),
            "turnId": str(row[2]),
            "attemptId": attempt_id,
            "causedByEventId": str(row[3]),
        }

    def events(self, context):
        return self.request(
            "GET", f"/executions/{context['rootExecutionId']}/events?limit=500"
        )["events"]

    def progress(self, context):
        return self.request(
            "GET", f"/executions/{context['rootExecutionId']}/progress"
        )

    @staticmethod
    def signals(events, action):
        return block_e2e.ExactToolRepeatBlockInterruptionTests.signals(
            events, action
        )

    @staticmethod
    def guard(progress):
        return block_e2e.ExactToolRepeatBlockInterruptionTests.guard(progress)

    def reassign(self, context):
        return block_e2e.ExactToolRepeatBlockInterruptionTests.reassign(
            self, context
        )

    def assert_stale_attempt_is_rejected(self, context, grant_id):
        return block_e2e.ExactToolRepeatBlockInterruptionTests.assert_stale_attempt_is_rejected(
            self, context, grant_id
        )

    def assert_ai_train_reconstructs(self, context, expected_status):
        return block_e2e.ExactToolRepeatBlockInterruptionTests.assert_ai_train_reconstructs(
            self, context, expected_status
        )

    def interrupt(self, context, phase, **extra):
        process = subprocess.run(
            [sys.executable, "-c", _CHILD_CODE],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "MVP12_EXECUTION_CONTEXT": json.dumps(context),
                "MVP12_INTERRUPT_PHASE": phase,
                "WORKER_ID": f"mvp12-{phase}",
                **extra,
            },
            check=False,
        )
        self.assertEqual(process.returncode, 91)

    def prepare_terminal_candidate(self):
        context = self.create_execution()
        self.interrupt(context, "after_block")
        progress = self.progress(context)
        grant_id = next(iter(progress["ledger"]["operationBudget"]["grants"]))
        guard = self.guard(progress)
        self.assertTrue(guard["blockResultPending"])
        self.assertEqual(guard["blocks"], 1)
        self.interrupt(context, "after_application", MVP12_GRANT_ID=grant_id)
        guard = self.guard(self.progress(context))
        self.assertFalse(guard["blockResultPending"])
        self.assertTrue(guard["blockResultAppliedToOperationId"])
        return context, grant_id

    @staticmethod
    def terminal_body(context, grant_id):
        return {
            "executionId": context["executionId"],
            "loopId": context["executionId"],
            "grantId": grant_id,
            "operationId": str(uuid.uuid4()),
            "operationKind": "tool_call",
            "bucket": "tool",
            "phase": "agent_loop",
            "round": 3,
            "name": "collect_evidence",
            "executionAttemptId": context["attemptId"],
            "toolCallId": str(uuid.uuid4()),
            "operationFingerprint": canonical_tool_input_fingerprint(
                "collect_evidence", {"index": 1, "nested": {"alpha": "β"}}
            ),
            "operationFingerprintVersion": "canonical_tool_input_v1",
            "toolBatchSize": 1,
            "toolBatchIndex": 0,
        }

    @staticmethod
    def post_reservation(context, body):
        emitter = ExecutionEmitter(context)
        try:
            return emitter._post("progress/reservations", body)
        finally:
            emitter.close()

    def assert_terminal_state(self, context, operation_id):
        events = self.events(context)
        self.assertEqual(len(self.signals(events, "terminate")), 1)
        self.assertFalse(any(
            event.get("operationId") == operation_id
            and event["eventType"] in {
                "operation.started", "operation.finished", "source.observed"
            }
            for event in events
        ))
        rebuilt = self.progress(context)
        grant = next(iter(
            rebuilt["ledger"]["operationBudget"]["grants"].values()
        ))
        self.assertEqual(grant["usage"]["tool"]["consumed"], 2)
        self.assertEqual(grant["usage"]["closing"]["consumed"], 0)
        self.assertEqual(self.guard(rebuilt)["terminations"], 1)
        return events

    def finalize_partial(self, context, grant_id, events):
        with self.database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {EXECUTIONS_TABLE} SET attempt_id = %s WHERE execution_id = %s",
                (context["attemptId"], context["executionId"]),
            )
        completed = [
            event for event in events
            if event["eventType"] == "operation.finished"
            and event["payload"].get("operationKind") == "tool_call"
            and event["payload"].get("resultSummary")
        ]
        started_by_operation = {
            event["operationId"]: event
            for event in events
            if event["eventType"] == "operation.started" and event.get("operationId")
        }
        completed_operations = [{
            "operationId": event["operationId"],
            "toolCallId": event["toolCallId"],
            "name": started_by_operation[event["operationId"]]["payload"]["name"],
            "summary": event["payload"]["resultSummary"],
        } for event in completed]
        reply = "\n".join([
            "I stopped this turn because it repeated an operation that had already been blocked.",
            "",
            "Completed work:",
            *[
                f"- Collect Evidence: {item['summary']}"
                for item in completed_operations
            ],
            "",
            "Continue in a new turn with different arguments or a different strategy.",
        ])
        result = {
            "reply": reply,
            "completionKind": "partial",
            "completionReason": "loop_detected",
            "completionSource": "runtime_template",
            "partialResult": {
                "version": "1",
                "trigger": "exact_tool_repeat_persisted",
                "loopId": context["executionId"],
                "grantId": grant_id,
                "executionAttemptId": context["attemptId"],
                "completedOperations": completed_operations,
                "pending": ["strategy_change"],
                "continuation": {
                    "kind": "new_turn",
                    "reason": "different_strategy_required",
                },
            },
        }
        emitter = ExecutionEmitter(context)
        try:
            emitter.record_final_message(reply, generation_source="runtime_template")
            emitter.flush_evidence()
        finally:
            emitter.close()
        with self.database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {EXECUTIONS_TABLE}
                SET result = %s::jsonb, phase = 'backend_finalization',
                    updated_at = now()
                WHERE execution_id = %s
                """,
                (json.dumps(result), context["executionId"]),
            )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with self.database_connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT status FROM {EXECUTIONS_TABLE} WHERE execution_id = %s",
                    (context["executionId"],),
                )
                status = cursor.fetchone()[0]
            if status == "completed":
                return
            if status == "failed":
                self.fail("backend rejected the durable loop partial")
            time.sleep(0.25)
        self.fail("backend did not finalize the durable loop partial")

    def test_three_cuts_replay_fencing_and_partial_reconstruction(self):
        context, grant_id = self.prepare_terminal_candidate()
        body = self.terminal_body(context, grant_id)
        self.interrupt(
            context,
            "after_terminate",
            MVP12_GRANT_ID=grant_id,
            MVP12_TERMINAL_BODY=json.dumps(body),
        )
        replayed = self.post_reservation(context, body)
        self.assertFalse(replayed["granted"])
        self.assertEqual(
            replayed["reservation"]["reason"],
            "immediate_exact_tool_repeat_terminated",
        )
        events = self.assert_terminal_state(context, body["operationId"])
        self.reassign(context)
        self.assert_stale_attempt_is_rejected(context, grant_id)
        self.finalize_partial(context, grant_id, events)
        self.assert_ai_train_reconstructs(context, "terminated_partial")

    def test_shared_identity_race_commits_one_terminal_decision(self):
        context, grant_id = self.prepare_terminal_candidate()
        body = self.terminal_body(context, grant_id)
        barrier = Barrier(2)

        def race(_index):
            barrier.wait()
            return self.post_reservation(context, body)

        with ThreadPoolExecutor(max_workers=2) as pool:
            decisions = list(pool.map(race, range(2)))
        self.assertTrue(all(not item["granted"] for item in decisions))
        self.assertEqual(len({item["eventId"] for item in decisions}), 1)
        self.assertEqual(
            len({
                item["reservation"]["reservationId"] for item in decisions
            }),
            1,
        )
        self.assert_terminal_state(context, body["operationId"])


if __name__ == "__main__":
    unittest.main()
