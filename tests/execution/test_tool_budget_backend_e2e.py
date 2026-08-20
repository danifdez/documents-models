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
from lib.execution import ExecutionEmitter, ToolBudgetDenied
from lib.execution.progress import ProgressLoopContext


_CHILD_CODE = r'''
import json
import os
import uuid

from lib.execution.emitter import ExecutionEmitter
from lib.execution.progress import ProgressLoopContext

context = json.loads(os.environ["MVP06_EXECUTION_CONTEXT"])
mode = os.environ["MVP06_INTERRUPT_MODE"]
emitter = ExecutionEmitter(context)
progress = ProgressLoopContext.start(
    emitter,
    agent_name="mvp06-interruption",
    loop_kind="top_level",
    max_rounds=1,
    max_output_repairs=0,
    forced_finalization_available=True,
    max_tokens_per_inference=64,
    max_tool_calls=1,
)
if mode == "after_reservation":
    emitter.reserve_operation_budget(
        grant_id=progress.grant_id,
        loop_id=progress.loop_id,
        operation_id=str(uuid.uuid4()),
        operation_kind="tool_call",
        bucket="tool",
        phase="agent_loop",
        round=1,
        name="first_value",
        tool_call_id=str(uuid.uuid4()),
    )
else:
    emitter.start_tool(
        "first_value",
        {},
        "provider-tool-call-1",
        progress.trace(round=1, phase="agent_loop"),
    )
    emitter.flush_evidence()
os._exit(91)
'''


@unittest.skipUnless(
    os.environ.get("RUN_TOOL_BUDGET_BACKEND_E2E") == "1",
    "set RUN_TOOL_BUDGET_BACKEND_E2E=1 with the isolated backend running",
)
class ToolBudgetInterruptionTests(unittest.TestCase):
    backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:3000").rstrip("/")

    def request(self, method, path, body=None):
        value = urllib.request.Request(
            f"{self.backend_url}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Workspace-Id": "mvp06-interruption",
            },
        )
        with urllib.request.urlopen(value, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def create_execution(self):
        assistant = self.request(
            "POST",
            "/assistants",
            {"name": "MVP 06 interruption", "systemPrompt": "Validation fixture"},
        )
        execution = self.request(
            "POST",
            f"/assistants/{assistant['id']}/messages",
            {"content": "MVP 06 interruption probe"},
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

    def database_connection(self):
        return psycopg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            autocommit=True,
        )

    def interrupt(self, mode):
        context = self.create_execution()
        environment = {
            **os.environ,
            "MVP06_EXECUTION_CONTEXT": json.dumps(context),
            "MVP06_INTERRUPT_MODE": mode,
            "WORKER_ID": f"mvp06-{mode}",
        }
        process = subprocess.run(
            [sys.executable, "-c", _CHILD_CODE],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env=environment,
            check=False,
        )
        self.assertEqual(process.returncode, 91)
        progress = self.request(
            "GET", f"/executions/{context['rootExecutionId']}/progress"
        )
        events = self.request(
            "GET", f"/executions/{context['rootExecutionId']}/events?limit=100"
        )["events"]
        return context, progress, events

    def assert_capacity_is_not_recovered(self, context, events, progress):
        grant = next(iter(progress["ledger"]["operationBudget"]["grants"].values()))
        next_attempt = str(uuid.uuid4())
        with self.database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {EXECUTIONS_TABLE} SET attempt_id = %s WHERE execution_id = %s",
                (next_attempt, context["executionId"]),
            )
        resumed = {
            **context,
            "attemptId": next_attempt,
            "causedByEventId": events[-1]["eventId"],
        }
        emitter = ExecutionEmitter(resumed)
        with self.assertRaisesRegex(
            ToolBudgetDenied, "tool_budget_hard_limit_reached"
        ):
            emitter.reserve_operation_budget(
                grant_id=grant["grantId"],
                loop_id=context["executionId"],
                operation_id=str(uuid.uuid4()),
                operation_kind="tool_call",
                bucket="tool",
                phase="agent_loop",
                round=1,
                name="second_value",
                tool_call_id=str(uuid.uuid4()),
            )

    def test_exit_after_reservation_keeps_the_slot_reserved(self):
        context, progress, events = self.interrupt("after_reservation")
        usage = next(iter(
            progress["ledger"]["operationBudget"]["grants"].values()
        ))["usage"]["tool"]
        self.assertEqual(
            usage,
            {
                "granted": 1,
                "reserved": 1,
                "consumed": 0,
                "available": 0,
                "softLimit": 0,
                "softLimitReached": False,
            },
        )
        self.assertFalse(any(
            event["eventType"] == "operation.started"
            and event["payload"].get("operationKind") == "tool_call"
            for event in events
        ))
        self.assert_capacity_is_not_recovered(context, events, progress)

    def test_exit_after_start_keeps_the_slot_consumed_and_tool_unfinished(self):
        context, progress, events = self.interrupt("after_start")
        usage = next(iter(
            progress["ledger"]["operationBudget"]["grants"].values()
        ))["usage"]["tool"]
        self.assertEqual(
            usage,
            {
                "granted": 1,
                "reserved": 0,
                "consumed": 1,
                "available": 0,
                "softLimit": 0,
                "softLimitReached": False,
            },
        )
        self.assertEqual(
            progress["ledger"]["operations"]["tool_call"],
            {"started": 1, "finished": 0, "unfinished": 1, "failed": 0},
        )
        self.assert_capacity_is_not_recovered(context, events, progress)

    def test_soft_limit_signal_survives_a_new_execution_attempt(self):
        context = self.create_execution()
        emitter = ExecutionEmitter(context)
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="mvp07-soft-limit",
            loop_kind="top_level",
            max_rounds=2,
            max_output_repairs=0,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
            max_tool_calls=6,
            tool_call_soft_limit=4,
        )
        last_handle = None
        for index in range(4):
            last_handle = emitter.start_tool(
                "lookup_value",
                {"index": index},
                f"provider-tool-call-{index}",
                progress.trace(round=1, phase="agent_loop"),
            )
            emitter.flush_evidence()

        self.assertIsNotNone(last_handle.soft_limit_signal)
        projected = self.request(
            "GET", f"/executions/{context['rootExecutionId']}/progress"
        )
        events = self.request(
            "GET", f"/executions/{context['rootExecutionId']}/events?limit=100"
        )["events"]
        usage = next(iter(
            projected["ledger"]["operationBudget"]["grants"].values()
        ))["usage"]["tool"]
        self.assertEqual(usage["available"], 2)
        self.assertEqual(usage["softLimit"], 4)
        self.assertTrue(usage["softLimitReached"])
        self.assertEqual(sum(
            event["payload"].get("kind") == "budget_soft_limit_reached"
            for event in events
        ), 1)

        next_attempt = str(uuid.uuid4())
        with self.database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {EXECUTIONS_TABLE} SET attempt_id = %s WHERE execution_id = %s",
                (next_attempt, context["executionId"]),
            )
        resumed = ExecutionEmitter({
            **context,
            "attemptId": next_attempt,
            "causedByEventId": events[-1]["eventId"],
        })
        recovered = ProgressLoopContext.start(
            resumed,
            agent_name="mvp07-soft-limit",
            loop_kind="top_level",
            max_rounds=2,
            max_output_repairs=0,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
            max_tool_calls=6,
            tool_call_soft_limit=4,
        )
        self.assertTrue(recovered.tool_soft_limit_reached)
        self.assertTrue(recovered.soft_limit_warning_pending)
        self.assertEqual(recovered.tool_budget_available, 2)


if __name__ == "__main__":
    unittest.main()
