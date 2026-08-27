import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from lib.execution.handler_process_pool import (
    HandlerPreempted,
    HandlerProcessFailed,
    HandlerProcessPool,
)

_PROCESS_EXECUTIONS = 0


def _process_executor(
    assignment,
    artifacts,
    output_artifacts=None,
):
    global _PROCESS_EXECUTIONS
    _PROCESS_EXECUTIONS += 1
    mode = assignment.get("mode")
    if mode == "block":
        Path(assignment["markerPath"]).write_text(
            str(os.getpid()), encoding="utf-8"
        )
        while True:
            time.sleep(0.1)
    if mode == "wait":
        Path(assignment["markerPath"]).write_text(
            str(os.getpid()), encoding="utf-8"
        )
        release = Path(assignment["releasePath"])
        while not release.exists():
            time.sleep(0.01)
    if mode == "crash":
        os._exit(7)
    return {
        "pid": os.getpid(),
        "executionCount": _PROCESS_EXECUTIONS,
        "artifact": artifacts.get("source", b"").decode("utf-8"),
    }


class HandlerProcessPoolTest(unittest.TestCase):
    def test_executes_a_registered_handler_in_the_isolated_process(self):
        pool = HandlerProcessPool(1)
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "service",
            "work": {
                "taskType": "detect-language",
                "payload": {
                    "samples": ["This sentence is written in English."]
                },
            },
        }
        try:
            result, output_artifacts = pool.execute(
                assignment,
                {},
                threading.Event(),
            )

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(
                result["output"]["value"]["results"][0]["language"],
                "en",
            )
            self.assertEqual(output_artifacts, [])
        finally:
            pool.close()

    def test_reuses_a_healthy_process_between_assignments(self):
        pool = HandlerProcessPool(1, executor=_process_executor)
        try:
            first, _ = pool.execute(
                {"mode": "complete"},
                {"source": b"first"},
                threading.Event(),
            )
            second, _ = pool.execute(
                {"mode": "complete"},
                {"source": b"second"},
                threading.Event(),
            )

            self.assertEqual(first["pid"], second["pid"])
            self.assertEqual(first["executionCount"], 1)
            self.assertEqual(second["executionCount"], 2)
            self.assertEqual(second["artifact"], "second")
        finally:
            pool.close()

    def test_runs_independent_slots_concurrently(self):
        pool = HandlerProcessPool(2, executor=_process_executor)
        results = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markers = [root / "first.pid", root / "second.pid"]
            releases = [root / "first.release", root / "second.release"]

            def execute(index):
                result, _ = pool.execute(
                    {
                        "mode": "wait",
                        "markerPath": str(markers[index]),
                        "releasePath": str(releases[index]),
                    },
                    {},
                    threading.Event(),
                )
                results.append(result)

            threads = [
                threading.Thread(target=execute, args=(index,))
                for index in range(2)
            ]
            try:
                for thread in threads:
                    thread.start()
                deadline = time.monotonic() + 3
                while (
                    not all(marker.exists() for marker in markers)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)

                self.assertTrue(all(marker.exists() for marker in markers))
                self.assertEqual(
                    len(
                        {
                            marker.read_text(encoding="utf-8")
                            for marker in markers
                        }
                    ),
                    2,
                )
                for release in releases:
                    release.write_text("ready", encoding="utf-8")
                for thread in threads:
                    thread.join(timeout=3)
                self.assertTrue(all(not thread.is_alive() for thread in threads))
                self.assertEqual(len(results), 2)
            finally:
                for release in releases:
                    release.write_text("ready", encoding="utf-8")
                for thread in threads:
                    thread.join(timeout=3)
                pool.close()

    def test_terminates_and_replaces_a_cancelled_handler_process(self):
        pool = HandlerProcessPool(1, executor=_process_executor)
        cancellation = threading.Event()
        outcome = {}
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "handler.pid"

            def execute_blocking_handler():
                try:
                    pool.execute(
                        {"mode": "block", "markerPath": str(marker)},
                        {},
                        cancellation,
                    )
                except Exception as error:
                    outcome["error"] = error

            thread = threading.Thread(target=execute_blocking_handler)
            thread.start()
            try:
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists())
                cancelled_pid = int(marker.read_text(encoding="utf-8"))

                cancellation.set()
                thread.join(timeout=3)

                self.assertFalse(thread.is_alive())
                self.assertIsInstance(outcome.get("error"), HandlerPreempted)
                replacement, _ = pool.execute(
                    {"mode": "complete"},
                    {},
                    threading.Event(),
                )
                self.assertNotEqual(replacement["pid"], cancelled_pid)
                self.assertEqual(replacement["executionCount"], 1)
            finally:
                cancellation.set()
                thread.join(timeout=3)
                pool.close()

    def test_replaces_a_handler_process_that_crashes(self):
        pool = HandlerProcessPool(1, executor=_process_executor)
        try:
            with self.assertRaises(HandlerProcessFailed):
                pool.execute(
                    {"mode": "crash"},
                    {},
                    threading.Event(),
                )

            recovered, _ = pool.execute(
                {"mode": "complete"},
                {},
                threading.Event(),
            )
            self.assertEqual(recovered["executionCount"], 1)
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
