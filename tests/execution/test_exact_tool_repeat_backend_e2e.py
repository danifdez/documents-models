import json
import os
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import psycopg

from config import (
    EXECUTIONS_TABLE,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from lib.execution import ExecutionEmitter


def ensure_ingest_token():
    if os.environ.get("EXECUTION_INGEST_TOKEN"):
        return
    env_path = Path(__file__).resolve().parents[3] / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("EXECUTION_INGEST_TOKEN="):
            os.environ["EXECUTION_INGEST_TOKEN"] = line.split("=", 1)[1]
            return
    raise RuntimeError("EXECUTION_INGEST_TOKEN is not configured for the profile")


_CHILD_CODE = r'''
import json
import os
import uuid

from lib.execution.emitter import (
    ExecutionEmitter,
    canonical_tool_input_fingerprint,
)
from lib.execution.progress import ProgressLoopContext

context = json.loads(os.environ["MVP10_EXECUTION_CONTEXT"])
mode = os.environ["MVP10_INTERRUPT_MODE"]
emitter = ExecutionEmitter(context)
progress = ProgressLoopContext.start(
    emitter,
    agent_name="mvp10-interruption",
    loop_kind="top_level",
    max_rounds=4,
    normal_inference_soft_limit=0,
    max_output_repairs=0,
    forced_finalization_available=True,
    max_tokens_per_inference=64,
    max_tool_calls=4,
    tool_call_soft_limit=0,
    exact_tool_repeat_warning=True,
)
arguments = {"index": 1, "nested": {"alpha": "β"}}
first = emitter.start_tool(
    "collect_evidence",
    arguments,
    "provider-tool-call-1",
    progress.trace(round=1, phase="agent_loop"),
    repeat_comparable=True,
)
emitter.flush_evidence()
first_source = emitter.observe_tool_result(
    first,
    {"validationMarker": "MVP10-CUT-DURABLE"},
)
emitter.finish_tool(
    first,
    {"validationMarker": "MVP10-CUT-DURABLE"},
    first_source,
    result_summary="Evidence collected: MVP10-CUT-DURABLE",
    result_summary_kind="leaf_tool",
)
emitter.flush_evidence()

fingerprint = canonical_tool_input_fingerprint("collect_evidence", arguments)
repeat_operation_id = str(uuid.uuid4())
repeat_tool_call_id = str(uuid.uuid4())
reservation_args = {
    "grant_id": progress.grant_id,
    "loop_id": progress.loop_id,
    "operation_id": repeat_operation_id,
    "operation_kind": "tool_call",
    "bucket": "tool",
    "phase": "agent_loop",
    "round": 2,
    "name": "collect_evidence",
    "tool_call_id": repeat_tool_call_id,
    "operation_fingerprint": fingerprint,
    "operation_fingerprint_version": "canonical_tool_input_v1",
}
repeat_reservation = emitter.reserve_operation_budget(**reservation_args)
repeat_retry = emitter.reserve_operation_budget(**reservation_args)
assert repeat_retry["reservationId"] == repeat_reservation["reservationId"]
assert repeat_retry["_loopGuardSignal"] == repeat_reservation["_loopGuardSignal"]
if mode == "after_signal":
    os._exit(91)

repeat = emitter.start_operation(
    "tool_call",
    "collect_evidence",
    extra_payload={
        "inputSummary": arguments,
        "providerToolCallId": "provider-tool-call-2",
        **progress.trace(round=2, phase="agent_loop"),
        "budgetGrantId": repeat_reservation["grantId"],
        "budgetReservationId": repeat_reservation["reservationId"],
        "budgetBucket": repeat_reservation["bucket"],
        "executionAttemptId": repeat_reservation["executionAttemptId"],
        "operationFingerprint": fingerprint,
        "operationFingerprintVersion": "canonical_tool_input_v1",
    },
    tool_call_id=repeat_tool_call_id,
    operation_id=repeat_operation_id,
)
emitter.flush_evidence()
repeat_source = emitter.observe_tool_result(
    repeat,
    {"validationMarker": "MVP10-CUT-DURABLE"},
)
emitter.finish_tool(
    repeat,
    {"validationMarker": "MVP10-CUT-DURABLE"},
    repeat_source,
    result_summary="Evidence collected again: MVP10-CUT-DURABLE",
    result_summary_kind="leaf_tool",
)
emitter.flush_evidence()
if mode == "after_repeat_finish":
    os._exit(91)

inference_operation_id = str(uuid.uuid4())
inference_args = {
    "grant_id": progress.grant_id,
    "loop_id": progress.loop_id,
    "operation_id": inference_operation_id,
    "operation_kind": "inference",
    "bucket": "normal",
    "phase": "agent_loop",
    "round": 3,
    "name": "chat_with_tools",
}
inference_reservation = emitter.reserve_operation_budget(**inference_args)
inference_retry = emitter.reserve_operation_budget(**inference_args)
assert inference_retry["reservationId"] == inference_reservation["reservationId"]
assert inference_retry["_guardState"]["warningPending"] is True
os._exit(91)
'''


@unittest.skipUnless(
    os.environ.get("RUN_EXACT_TOOL_REPEAT_BACKEND_E2E") == "1",
    "set RUN_EXACT_TOOL_REPEAT_BACKEND_E2E=1 with the isolated backend running",
)
class ExactToolRepeatInterruptionTests(unittest.TestCase):
    backend_url = os.environ.get(
        "BACKEND_URL", "http://127.0.0.1:3000"
    ).rstrip("/")

    @classmethod
    def setUpClass(cls):
        ensure_ingest_token()

    def request(self, method, path, body=None):
        value = urllib.request.Request(
            f"{self.backend_url}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Workspace-Id": "mvp10-interruption",
            },
        )
        with urllib.request.urlopen(value, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def database_connection(self):
        return psycopg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            autocommit=True,
        )

    def create_execution(self):
        assistant = self.request(
            "POST",
            "/assistants",
            {"name": "MVP 10 interruption", "systemPrompt": "Validation fixture"},
        )
        execution = self.request(
            "POST",
            f"/assistants/{assistant['id']}/messages",
            {"content": "MVP 10 interruption probe"},
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
            "GET",
            f"/executions/{context['rootExecutionId']}/events?limit=500",
        )["events"]

    def progress(self, context):
        return self.request(
            "GET",
            f"/executions/{context['rootExecutionId']}/progress",
        )

    def interrupt(self, mode):
        context = self.create_execution()
        process = subprocess.run(
            [sys.executable, "-c", _CHILD_CODE],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env={
                **os.environ,
                "MVP10_EXECUTION_CONTEXT": json.dumps(context),
                "MVP10_INTERRUPT_MODE": mode,
                "WORKER_ID": f"mvp10-{mode}",
            },
            check=False,
        )
        self.assertEqual(process.returncode, 91)
        return context, self.events(context), self.progress(context)

    @staticmethod
    def signals(events):
        return [
            event
            for event in events
            if event["eventType"] == "progress.reported"
            and event["payload"].get("kind") == "loop_guard_triggered"
        ]

    @staticmethod
    def guard(progress):
        values = (progress["ledger"].get("loopGuards") or {}).values()
        return next(iter(values))["exactToolRepeat"]

    def reassign(self, context):
        next_attempt = str(uuid.uuid4())
        with self.database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {EXECUTIONS_TABLE} SET attempt_id = %s WHERE execution_id = %s",
                (next_attempt, context["executionId"]),
            )
        events = self.events(context)
        return {
            **context,
            "attemptId": next_attempt,
            "causedByEventId": events[-1]["eventId"],
        }

    def assert_stale_attempt_is_rejected(self, context, grant_id, loop_id):
        stale = ExecutionEmitter(context)
        with self.assertRaisesRegex(Exception, "budget_reservation_failed"):
            stale.reserve_operation_budget(
                grant_id=grant_id,
                loop_id=loop_id,
                operation_id=str(uuid.uuid4()),
                operation_kind="inference",
                bucket="normal",
                phase="agent_loop",
                round=4,
                name="chat_with_tools",
            )

    def assert_ai_train_reconstructs(self, context, expected_status):
        root = os.environ.get("AI_TRAIN_ROOT")
        if not root:
            self.skipTest("set AI_TRAIN_ROOT to validate the exported bundle")
        bundle = self.request(
            "GET", f"/executions/{context['rootExecutionId']}/bundle"
        )
        script = (
            "import json,sys; "
            "from harness.execution_bundle import import_bundle,project_bundle; "
            "value=project_bundle(import_bundle(json.load(sys.stdin))); "
            "items=value['execution']['progress']['loopGuardEvaluation']"
            "['exactToolRepeat']; "
            "print(items[0]['status'])"
        )
        python = Path(root) / ".venv" / "bin" / "python"
        process = subprocess.run(
            [str(python), "-c", script],
            cwd=root,
            input=json.dumps(bundle),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), expected_status)

    def finalize_execution(self, context):
        with self.database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {EXECUTIONS_TABLE}
                SET result = %s::jsonb, phase = 'backend_finalization',
                    updated_at = now()
                WHERE execution_id = %s
                """,
                (
                    json.dumps({"reply": "MVP10-CUT-RECOVERED"}),
                    context["executionId"],
                ),
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
            time.sleep(0.25)
        self.fail("backend did not finalize the interrupted execution")

    def consume_warning_after_reassignment(
        self,
        context,
        *,
        repeat_after_warning=False,
    ):
        progress = self.progress(context)
        grant_id = next(iter(
            progress["ledger"]["operationBudget"]["grants"]
        ))
        loop_id = context["executionId"]
        resumed_context = self.reassign(context)
        self.assert_stale_attempt_is_rejected(context, grant_id, loop_id)
        resumed = ExecutionEmitter(resumed_context)
        self.addCleanup(resumed.close)
        request = {"messages": [{"role": "user", "content": "finish"}]}
        handle = resumed.start_inference(
            "chat_with_tools",
            request,
            {
                "loopId": loop_id,
                "agentName": "mvp10-interruption",
                "loopKind": "top_level",
                "maxRounds": 4,
                "round": 4,
                "phase": "agent_loop",
                "budgetGrantId": grant_id,
            },
        )
        resumed.flush_evidence()
        resumed.finish_inference(
            handle,
            {"content": "MVP10-CUT-RECOVERED"},
            outcome="tool_requests" if repeat_after_warning else "final_text",
        )
        resumed.flush_evidence()
        if repeat_after_warning:
            repeated = resumed.start_tool(
                "collect_evidence",
                {"index": 1, "nested": {"alpha": "β"}},
                "provider-tool-call-after-warning",
                {
                    "loopId": loop_id,
                    "agentName": "mvp10-interruption",
                    "loopKind": "top_level",
                    "maxRounds": 4,
                    "round": 4,
                    "phase": "agent_loop",
                    "budgetGrantId": grant_id,
                },
                repeat_comparable=True,
            )
            resumed.flush_evidence()
            source = resumed.observe_tool_result(
                repeated,
                {"validationMarker": "MVP10-PERSISTED"},
            )
            resumed.finish_tool(
                repeated,
                {"validationMarker": "MVP10-PERSISTED"},
                source,
                result_summary="Evidence repeated after warning",
                result_summary_kind="leaf_tool",
            )
            resumed.flush_evidence()
        self.assertEqual(sum(
            message.get("content", "").startswith(
                "An exact tool call was repeated"
            )
            for message in request["messages"]
        ), 1)
        rebuilt = self.progress(context)
        self.assertFalse(self.guard(rebuilt)["warningPending"])
        rebuilt_events = self.events(context)
        self.assertEqual(len(self.signals(rebuilt_events)), 1)
        self.assertEqual(sum(
            event["eventType"] == "operation.started"
            and event["payload"].get("loopGuardWarningApplied") is True
            for event in rebuilt_events
        ), 1)
        self.finalize_execution(context)
        self.assert_ai_train_reconstructs(
            context,
            "persisted" if repeat_after_warning else "recovered",
        )

    def test_exit_after_signal_keeps_one_pending_warning(self):
        context, events, progress = self.interrupt("after_signal")
        self.assertEqual(len(self.signals(events)), 1)
        self.assertTrue(self.guard(progress)["warningPending"])
        self.assertEqual(sum(
            event["eventType"] == "operation.started"
            and event["payload"].get("name") == "collect_evidence"
            for event in events
        ), 1)
        self.consume_warning_after_reassignment(context)

    def test_exit_after_repeat_finish_keeps_the_warning_and_both_results(self):
        context, events, progress = self.interrupt("after_repeat_finish")
        self.assertEqual(len(self.signals(events)), 1)
        self.assertTrue(self.guard(progress)["warningPending"])
        self.assertEqual(sum(
            event["eventType"] == "operation.finished"
            and event["payload"].get("operationKind") == "tool_call"
            and event["payload"].get("status") == "succeeded"
            for event in events
        ), 2)
        self.consume_warning_after_reassignment(context)

    def test_exit_after_warning_reservation_does_not_consume_it(self):
        context, events, progress = self.interrupt("after_inference_reservation")
        self.assertEqual(len(self.signals(events)), 1)
        self.assertTrue(self.guard(progress)["warningPending"])
        reservations = progress["ledger"]["operationBudget"]["reservations"].values()
        self.assertEqual(sum(
            value.get("operationKind") == "inference"
            and value.get("bucket") == "normal"
            and value.get("status") == "reserved"
            for value in reservations
        ), 1)
        self.assertFalse(any(
            event["eventType"] == "operation.started"
            and event["payload"].get("operationKind") == "inference"
            for event in events
        ))
        self.consume_warning_after_reassignment(context)

    def test_ai_train_classifies_a_post_warning_repeat_as_persisted(self):
        context, events, progress = self.interrupt("after_repeat_finish")
        self.assertEqual(len(self.signals(events)), 1)
        self.assertTrue(self.guard(progress)["warningPending"])
        self.consume_warning_after_reassignment(
            context,
            repeat_after_warning=True,
        )


if __name__ == "__main__":
    unittest.main()
