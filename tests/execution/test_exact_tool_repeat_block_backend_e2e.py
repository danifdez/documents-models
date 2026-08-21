import json
import os
import subprocess
import sys
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
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
from lib.execution.emitter import canonical_tool_input_fingerprint


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

from lib.execution import ExecutionEmitter
from lib.execution.emitter import canonical_tool_input_fingerprint
from lib.execution.progress import ProgressLoopContext

context = json.loads(os.environ["MVP11_EXECUTION_CONTEXT"])
mode = os.environ["MVP11_INTERRUPT_MODE"]
emitter = ExecutionEmitter(context)
progress = ProgressLoopContext.start(
    emitter,
    agent_name="mvp11-interruption",
    loop_kind="top_level",
    max_rounds=4,
    normal_inference_soft_limit=0,
    max_output_repairs=0,
    forced_finalization_available=True,
    max_tokens_per_inference=64,
    max_tool_calls=4,
    tool_call_soft_limit=0,
    exact_tool_repeat_warning=True,
    exact_tool_repeat_block_after_warning=True,
)
arguments = {"index": 1, "nested": {"alpha": "β"}}
for round_number in (1, 2):
    handle = emitter.start_tool(
        "collect_evidence",
        arguments,
        f"provider-tool-call-{round_number}",
        progress.trace(round=round_number, phase="agent_loop"),
        repeat_comparable=True,
    )
    emitter.flush_evidence()
    result = {"validationMarker": "MVP11-CUT-DURABLE"}
    source = emitter.observe_tool_result(handle, result)
    emitter.finish_tool(
        handle,
        result,
        source,
        result_summary="Evidence collected: MVP11-CUT-DURABLE",
        result_summary_kind="leaf_tool",
    )
    emitter.flush_evidence()

request = {"messages": [{"role": "user", "content": "continue"}]}
inference = emitter.start_inference(
    "chat_with_tools",
    request,
    progress.trace(round=3, phase="agent_loop"),
)
emitter.flush_evidence()
assert sum(
    message.get("content", "").startswith("An exact tool call was repeated")
    for message in request["messages"]
) == 1
emitter.finish_inference(
    inference,
    {"content": "request another tool"},
    outcome="tool_requests",
)
emitter.flush_evidence()
if mode == "after_warning":
    os._exit(91)

operation_id = str(uuid.uuid4())
tool_call_id = str(uuid.uuid4())
fingerprint = canonical_tool_input_fingerprint("collect_evidence", arguments)
body = {
    "executionId": context["executionId"],
    "loopId": progress.loop_id,
    "grantId": progress.grant_id,
    "operationId": operation_id,
    "operationKind": "tool_call",
    "bucket": "tool",
    "phase": "agent_loop",
    "round": 3,
    "name": "collect_evidence",
    "executionAttemptId": context["attemptId"],
    "toolCallId": tool_call_id,
    "operationFingerprint": fingerprint,
    "operationFingerprintVersion": "canonical_tool_input_v1",
}
blocked = emitter._post("progress/reservations", body)
replayed = emitter._post("progress/reservations", body)
assert blocked["granted"] is False
assert blocked["reservation"]["reason"] == "immediate_exact_tool_repeat_blocked"
assert replayed["reservation"]["reservationId"] == blocked["reservation"]["reservationId"]
assert replayed["eventId"] == blocked["eventId"]
assert replayed["loopGuardSignal"] == blocked["loopGuardSignal"]
os._exit(91)
'''


@unittest.skipUnless(
    os.environ.get("RUN_EXACT_TOOL_REPEAT_BLOCK_BACKEND_E2E") == "1",
    "set RUN_EXACT_TOOL_REPEAT_BLOCK_BACKEND_E2E=1 with the backend running",
)
class ExactToolRepeatBlockInterruptionTests(unittest.TestCase):
    backend_url = os.environ.get(
        "BACKEND_URL", "http://127.0.0.1:3000"
    ).rstrip("/")

    @classmethod
    def setUpClass(cls):
        ensure_ingest_token()

    def request(self, method, path, body=None):
        import urllib.request

        value = urllib.request.Request(
            f"{self.backend_url}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Workspace-Id": "mvp11-interruption",
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
            {"name": "MVP 11 interruption", "systemPrompt": "Validation fixture"},
        )
        execution = self.request(
            "POST",
            f"/assistants/{assistant['id']}/messages",
            {"content": "MVP 11 interruption probe"},
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
            "GET", f"/executions/{context['rootExecutionId']}/progress"
        )

    def interrupt(self, mode):
        context = self.create_execution()
        process = subprocess.run(
            [sys.executable, "-c", _CHILD_CODE],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "MVP11_EXECUTION_CONTEXT": json.dumps(context),
                "MVP11_INTERRUPT_MODE": mode,
                "WORKER_ID": f"mvp11-{mode}",
            },
            check=False,
        )
        self.assertEqual(process.returncode, 91)
        return context

    @staticmethod
    def signals(events, action):
        return [
            event
            for event in events
            if event["eventType"] == "progress.reported"
            and event["payload"].get("kind") == "loop_guard_triggered"
            and (event["payload"].get("loopGuardSignal") or {}).get("action")
            == action
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
        return {**context, "attemptId": next_attempt}

    def assert_stale_attempt_is_rejected(self, context, grant_id):
        stale = ExecutionEmitter(context)
        self.addCleanup(stale.close)
        with self.assertRaisesRegex(Exception, "budget_reservation_failed"):
            stale.reserve_operation_budget(
                grant_id=grant_id,
                loop_id=context["executionId"],
                operation_id=str(uuid.uuid4()),
                operation_kind="inference",
                bucket="normal",
                phase="agent_loop",
                round=4,
                name="chat_with_tools",
            )

    def block_request(self, context, grant_id, operation_id, tool_call_id):
        emitter = ExecutionEmitter(context)
        self.addCleanup(emitter.close)
        body = {
            "executionId": context["executionId"],
            "loopId": context["executionId"],
            "grantId": grant_id,
            "operationId": operation_id,
            "operationKind": "tool_call",
            "bucket": "tool",
            "phase": "agent_loop",
            "round": 4,
            "name": "collect_evidence",
            "executionAttemptId": context["attemptId"],
            "toolCallId": tool_call_id,
            "operationFingerprint": canonical_tool_input_fingerprint(
                "collect_evidence", {"index": 1, "nested": {"alpha": "β"}}
            ),
            "operationFingerprintVersion": "canonical_tool_input_v1",
        }
        return emitter._post("progress/reservations", body), body

    def reserve_block(self, context, grant_id):
        operation_id = str(uuid.uuid4())
        blocked, body = self.block_request(
            context, grant_id, operation_id, str(uuid.uuid4())
        )
        emitter = ExecutionEmitter(context)
        self.addCleanup(emitter.close)
        replayed = emitter._post("progress/reservations", body)
        self.assertFalse(blocked["granted"])
        self.assertEqual(
            blocked["reservation"]["reason"],
            "immediate_exact_tool_repeat_blocked",
        )
        self.assertEqual(replayed["eventId"], blocked["eventId"])
        self.assertEqual(
            replayed["reservation"]["reservationId"],
            blocked["reservation"]["reservationId"],
        )
        self.assertEqual(replayed["loopGuardSignal"], blocked["loopGuardSignal"])
        return operation_id

    def close_after_reassignment(self, context, grant_id):
        emitter = ExecutionEmitter(context)
        self.addCleanup(emitter.close)
        request = {"messages": [{"role": "user", "content": "finish safely"}]}
        handle = emitter.start_inference(
            "forced_finalization",
            request,
            {
                "loopId": context["executionId"],
                "agentName": "mvp11-interruption",
                "loopKind": "top_level",
                "maxRounds": 4,
                "round": 4,
                "phase": "forced_finalization",
                "budgetGrantId": grant_id,
            },
        )
        emitter.flush_evidence()
        emitter.finish_inference(
            handle,
            {"content": "MVP11-CUT-CLOSED"},
            outcome="final_text",
        )
        emitter.record_final_message("MVP11-CUT-CLOSED")
        emitter.flush_evidence()

    def assert_durable_block(self, context, blocked_operation_id):
        events = self.events(context)
        progress = self.progress(context)
        self.assertEqual(len(self.signals(events, "warn")), 1)
        self.assertEqual(len(self.signals(events, "block")), 1)
        self.assertEqual(self.guard(progress)["blocks"], 1)
        self.assertFalse(self.guard(progress)["warningPending"])
        self.assertFalse(any(
            event.get("operationId") == blocked_operation_id
            and event["eventType"] in {
                "operation.started", "operation.finished", "source.observed"
            }
            for event in events
        ))
        grant = next(iter(progress["ledger"]["operationBudget"]["grants"].values()))
        self.assertEqual(grant["usage"]["tool"]["consumed"], 2)

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
            "['exactToolRepeat']; print(items[0]['status'])"
        )
        process = subprocess.run(
            [str(Path(root) / ".venv" / "bin" / "python"), "-c", script],
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
                (json.dumps({"reply": "MVP11-CUT-CLOSED"}), context["executionId"]),
            )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with self.database_connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT status FROM {EXECUTIONS_TABLE} WHERE execution_id = %s",
                    (context["executionId"],),
                )
                if cursor.fetchone()[0] == "completed":
                    return
            time.sleep(0.25)
        self.fail("backend did not finalize the interrupted execution")

    def test_cut_after_warning_preserves_state_and_blocks_after_restart(self):
        context = self.interrupt("after_warning")
        progress = self.progress(context)
        grant_id = next(iter(progress["ledger"]["operationBudget"]["grants"]))
        self.assertTrue(self.guard(progress)["warningIssued"])
        self.assertFalse(self.guard(progress)["warningPending"])
        blocked_operation_id = self.reserve_block(context, grant_id)
        self.assert_durable_block(context, blocked_operation_id)
        resumed_context = self.reassign(context)
        self.assert_stale_attempt_is_rejected(context, grant_id)
        self.close_after_reassignment(resumed_context, grant_id)
        self.finalize_execution(context)
        self.assert_ai_train_reconstructs(context, "closed")

    def test_cut_after_block_replays_once_and_closes_after_reassignment(self):
        context = self.interrupt("after_block")
        progress = self.progress(context)
        grant_id = next(iter(progress["ledger"]["operationBudget"]["grants"]))
        blocked = [
            value
            for value in progress["ledger"]["operationBudget"]["reservations"].values()
            if value.get("reason") == "immediate_exact_tool_repeat_blocked"
        ]
        self.assertEqual(len(blocked), 1)
        blocked_operation_id = blocked[0]["operationId"]
        self.assert_durable_block(context, blocked_operation_id)
        resumed_context = self.reassign(context)
        self.assert_stale_attempt_is_rejected(context, grant_id)
        self.close_after_reassignment(resumed_context, grant_id)
        self.finalize_execution(context)
        self.assert_ai_train_reconstructs(context, "closed")

    def test_concurrent_distinct_proposals_are_both_blocked(self):
        context = self.interrupt("after_warning")
        progress = self.progress(context)
        grant_id = next(iter(progress["ledger"]["operationBudget"]["grants"]))
        operation_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        with ThreadPoolExecutor(max_workers=2) as pool:
            decisions = list(pool.map(
                lambda operation_id: self.block_request(
                    context, grant_id, operation_id, str(uuid.uuid4())
                )[0],
                operation_ids,
            ))
        self.assertTrue(all(not decision["granted"] for decision in decisions))
        self.assertEqual(
            {
                decision["loopGuardSignal"]["triggeringOperationId"]
                for decision in decisions
            },
            set(operation_ids),
        )
        events = self.events(context)
        self.assertEqual(len(self.signals(events, "warn")), 1)
        self.assertEqual(len(self.signals(events, "block")), 2)
        self.assertFalse(any(
            event.get("operationId") in operation_ids
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
        self.assertEqual(self.guard(rebuilt)["blocks"], 2)
        resumed_context = self.reassign(context)
        self.assert_stale_attempt_is_rejected(context, grant_id)
        self.finalize_execution(resumed_context)
        self.assert_ai_train_reconstructs(context, "persisted")


if __name__ == "__main__":
    unittest.main()
