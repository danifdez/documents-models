import json
import os
import subprocess
import sys
import unittest
import urllib.request
import uuid


_CHILD_CODE = r'''
import json
import os

from lib.execution.emitter import ExecutionEmitter
from lib.execution.progress import ProgressLoopContext

context = json.loads(os.environ["MVP04_EXECUTION_CONTEXT"])
mode = os.environ["MVP04_INTERRUPT_MODE"]
emitter = ExecutionEmitter(context)
progress = ProgressLoopContext.start(
    emitter,
    agent_name="mvp04-e2e",
    loop_kind="top_level",
    max_rounds=1,
    max_output_repairs=0,
    forced_finalization_available=False,
    max_tokens_per_inference=64,
)
handle = emitter.start_inference(
    "direct_response",
    {"messages": []},
    progress.trace(round=1, phase="direct_response"),
)
emitter.flush_evidence()
if mode == "after_finish":
    emitter.finish_inference(
        handle,
        {"content": "durable"},
        outcome="final_text",
        metrics={"durationMs": 7, "promptTokens": 13, "generatedTokens": 5},
    )
    emitter.flush_evidence()
os._exit(91)
'''


@unittest.skipUnless(
    os.environ.get("RUN_PROGRESS_BACKEND_E2E") == "1",
    "set RUN_PROGRESS_BACKEND_E2E=1 with the isolated backend running",
)
class ProgressBackendInterruptionTests(unittest.TestCase):
    backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:3000").rstrip("/")

    def request(self, method, path, body=None):
        request = urllib.request.Request(
            f"{self.backend_url}{path}",
            data=(json.dumps(body).encode("utf-8") if body is not None else None),
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Workspace-Id": "mvp04-validation",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def create_execution(self):
        execution = self.request(
            "POST",
            "/executions",
            {"taskType": "search", "content": "MVP 04 interruption probe"},
        )
        return {
            "rootExecutionId": execution["rootExecutionId"],
            "executionId": execution["executionId"],
            "attemptId": str(uuid.uuid4()),
            "causedByEventId": execution["lastEventId"],
        }

    def interrupt(self, mode):
        context = self.create_execution()
        environment = {
            **os.environ,
            "MVP04_EXECUTION_CONTEXT": json.dumps(context),
            "MVP04_INTERRUPT_MODE": mode,
            "WORKER_ID": f"mvp04-{mode}",
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
        return progress, events

    def test_process_exit_after_started_keeps_an_unfinished_operation(self):
        progress, events = self.interrupt("after_start")

        self.assertEqual(progress["ledger"]["operations"]["inference"], {
            "started": 1,
            "finished": 0,
            "unfinished": 1,
            "failed": 0,
        })
        self.assertEqual(progress["ledger"]["completeness"], "partial")
        self.assertEqual(
            [event["eventType"] for event in events[-2:]],
            ["progress.reported", "operation.started"],
        )

    def test_process_exit_after_finished_keeps_tokens_and_completion(self):
        progress, events = self.interrupt("after_finish")

        self.assertEqual(progress["ledger"]["operations"]["inference"], {
            "started": 1,
            "finished": 1,
            "unfinished": 0,
            "failed": 0,
        })
        self.assertEqual(progress["ledger"]["promptTokens"]["total"], 13)
        self.assertEqual(progress["ledger"]["generatedTokens"]["total"], 5)
        self.assertEqual(progress["ledger"]["completeness"], "complete")
        self.assertEqual(events[-1]["eventType"], "operation.finished")


if __name__ == "__main__":
    unittest.main()
