import json
import os
import subprocess
import sys
import unittest
import urllib.request
import uuid

import psycopg

from config import (
    EXECUTIONS_TABLE,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)


_CHILD_CODE = r'''
import json
import os

from lib.execution.emitter import ExecutionEmitter
from lib.execution.progress import ProgressLoopContext

context = json.loads(os.environ["MVP09_EXECUTION_CONTEXT"])
mode = os.environ["MVP09_INTERRUPT_MODE"]
emitter = ExecutionEmitter(context)
progress = ProgressLoopContext.start(
    emitter,
    agent_name="mvp09-interruption",
    loop_kind="top_level",
    max_rounds=1,
    max_output_repairs=0,
    forced_finalization_available=True,
    max_tokens_per_inference=64,
    max_tool_calls=1,
)
inference = emitter.start_inference(
    "chat_with_tools",
    {"messages": [{"role": "user", "content": "Collect durable evidence"}]},
    progress.trace(round=1, phase="agent_loop"),
)
emitter.flush_evidence()
emitter.finish_inference(
    inference,
    {"content": "", "tool_calls": [{"id": "provider-tool-call-1"}]},
    outcome="tool_requests",
)
tool = emitter.start_tool(
    "collect_evidence",
    {"index": 1},
    "provider-tool-call-1",
    progress.trace(round=1, phase="agent_loop"),
)
emitter.flush_evidence()
source_event_id = emitter.observe_tool_result(
    tool,
    {"validationMarker": "MVP09-CUT-DURABLE"},
)
emitter.finish_tool(
    tool,
    {"validationMarker": "MVP09-CUT-DURABLE"},
    source_event_id,
    result_summary="Evidence collected: MVP09-CUT-DURABLE",
    result_summary_kind="leaf_tool",
)
emitter.flush_evidence()
if mode == "after_tool":
    os._exit(91)

closing = emitter.start_inference(
    "forced_finalization",
    {"messages": [{"role": "user", "content": "Produce the final answer"}]},
    progress.trace(
        round=1,
        phase="forced_finalization",
        extra={"reason": "step_budget_exhausted"},
    ),
)
emitter.flush_evidence()
emitter.finish_inference(
    closing,
    {"content": ""},
    outcome="invalid",
    reason="empty_model_response",
)
emitter.flush_evidence()
if mode == "after_closing":
    os._exit(91)

emitter.record_final_message(
    "Completed work:\n- Evidence collected: MVP09-CUT-DURABLE",
    generation_source="runtime_template",
)
emitter.flush_evidence()
os._exit(91)
'''


@unittest.skipUnless(
    os.environ.get("RUN_DETERMINISTIC_PARTIAL_BACKEND_E2E") == "1",
    "set RUN_DETERMINISTIC_PARTIAL_BACKEND_E2E=1 with the isolated backend running",
)
class DeterministicPartialInterruptionTests(unittest.TestCase):
    backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:3000").rstrip("/")

    def request(self, method, path, body=None):
        value = urllib.request.Request(
            f"{self.backend_url}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Workspace-Id": "mvp09-interruption",
            },
        )
        with urllib.request.urlopen(value, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

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
            {"name": "MVP 09 interruption", "systemPrompt": "Validation fixture"},
        )
        execution = self.request(
            "POST",
            f"/assistants/{assistant['id']}/messages",
            {"content": "MVP 09 interruption probe"},
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

    def interrupt(self, mode):
        context = self.create_execution()
        process = subprocess.run(
            [sys.executable, "-c", _CHILD_CODE],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env={
                **os.environ,
                "MVP09_EXECUTION_CONTEXT": json.dumps(context),
                "MVP09_INTERRUPT_MODE": mode,
                "WORKER_ID": f"mvp09-{mode}",
            },
            check=False,
        )
        self.assertEqual(process.returncode, 91)
        events = self.request(
            "GET", f"/executions/{context['rootExecutionId']}/events?limit=100"
        )["events"]
        return events

    @staticmethod
    def matching(events, event_type, operation_kind=None, outcome=None):
        return [
            event
            for event in events
            if event["eventType"] == event_type
            and (
                operation_kind is None
                or event["payload"].get("operationKind") == operation_kind
            )
            and (outcome is None or event["payload"].get("outcome") == outcome)
        ]

    def assert_durable_tool(self, events):
        finished = self.matching(events, "operation.finished", "tool_call")
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["payload"]["status"], "succeeded")
        self.assertEqual(finished[0]["payload"]["resultSummaryKind"], "leaf_tool")
        self.assertIn("MVP09-CUT-DURABLE", finished[0]["payload"]["resultSummary"])

    def runtime_messages(self, events):
        return [
            event
            for event in self.matching(events, "message.recorded")
            if event["payload"].get("generationSource") == "runtime_template"
        ]

    def test_exit_after_tool_keeps_the_eligible_leaf(self):
        events = self.interrupt("after_tool")
        self.assert_durable_tool(events)
        self.assertEqual(
            self.matching(events, "operation.finished", "inference", "invalid"),
            [],
        )
        self.assertEqual(self.runtime_messages(events), [])

    def test_exit_after_invalid_closing_keeps_the_trigger(self):
        events = self.interrupt("after_closing")
        self.assert_durable_tool(events)
        closings = self.matching(
            events, "operation.finished", "inference", "invalid"
        )
        self.assertEqual(len(closings), 1)
        self.assertEqual(
            closings[0]["payload"]["reason"],
            "empty_model_response",
        )
        self.assertEqual(self.runtime_messages(events), [])

    def test_exit_after_runtime_message_keeps_the_complete_partial_evidence(self):
        events = self.interrupt("after_message")
        self.assert_durable_tool(events)
        self.assertEqual(
            len(self.matching(events, "operation.finished", "inference", "invalid")),
            1,
        )
        messages = self.runtime_messages(events)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["actor"], {"type": "system"})
        self.assertEqual(
            messages[0]["payload"]["generationSource"],
            "runtime_template",
        )


if __name__ == "__main__":
    unittest.main()
