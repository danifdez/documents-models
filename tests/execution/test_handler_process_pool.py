import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lib.execution.handler_process_pool import (
    HandlerPreempted,
    HandlerProcessFailed,
    HandlerProcessPool,
    _terminate_process_tree,
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
    if mode == "temporary":
        descriptor, path = tempfile.mkstemp()
        os.close(descriptor)
        Path(path).write_text("temporary", encoding="utf-8")
        return {"pid": os.getpid(), "temporaryPath": path}
    if mode == "temporary-block":
        descriptor, path = tempfile.mkstemp()
        os.close(descriptor)
        Path(path).write_text("temporary", encoding="utf-8")
        Path(assignment["markerPath"]).write_text(path, encoding="utf-8")
        while True:
            time.sleep(0.1)
    if mode == "descendant":
        heartbeat = assignment["heartbeatPath"]
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib,sys,time\n"
                    "p = pathlib.Path(sys.argv[1])\n"
                    "while True:\n"
                    "    p.write_text(str(time.monotonic_ns()))\n"
                    "    time.sleep(0.02)\n"
                ),
                heartbeat,
            ]
        )
        Path(assignment["markerPath"]).write_text(
            str(child.pid), encoding="utf-8"
        )
        child.wait()
    return {
        "pid": os.getpid(),
        "executionCount": _PROCESS_EXECUTIONS,
        "artifact": artifacts.get("source", b"").decode("utf-8"),
    }


class HandlerProcessPoolTest(unittest.TestCase):
    def test_uses_taskkill_for_the_process_tree_on_windows(self):
        process = Mock()
        process.pid = 321
        with patch(
            "lib.execution.handler_process_pool.os.name",
            "nt",
        ), patch(
            "lib.execution.handler_process_pool.subprocess.run"
        ) as run:
            _terminate_process_tree(process, force=False)

        run.assert_called_once_with(
            ["taskkill", "/PID", "321", "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        )

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

    def test_removes_the_handler_temporary_directory_after_success(self):
        pool = HandlerProcessPool(1, executor=_process_executor)
        try:
            result, _ = pool.execute(
                {"mode": "temporary"},
                {},
                threading.Event(),
            )

            temporary_path = Path(result["temporaryPath"])
            self.assertFalse(temporary_path.exists())
            self.assertFalse(temporary_path.parent.exists())
        finally:
            pool.close()

    def test_releases_the_slot_when_temporary_directory_creation_fails(self):
        pool = HandlerProcessPool(1, executor=_process_executor)
        try:
            with patch(
                "lib.execution.handler_process_pool.tempfile.mkdtemp",
                side_effect=OSError("no temporary storage"),
            ):
                with self.assertRaisesRegex(
                    HandlerProcessFailed,
                    "Could not create handler temporary directory",
                ):
                    pool.execute({}, {}, threading.Event())

            result, _ = pool.execute({}, {}, threading.Event())
            self.assertIn("pid", result)
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

    def test_removes_temporary_files_after_preemption(self):
        pool = HandlerProcessPool(1, executor=_process_executor)
        cancellation = threading.Event()
        outcome = {}
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "temporary-path.txt"

            def execute_blocking_handler():
                try:
                    pool.execute(
                        {
                            "mode": "temporary-block",
                            "markerPath": str(marker),
                        },
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
                temporary_path = Path(marker.read_text(encoding="utf-8"))
                self.assertTrue(temporary_path.exists())

                cancellation.set()
                thread.join(timeout=3)

                self.assertFalse(thread.is_alive())
                self.assertIsInstance(outcome.get("error"), HandlerPreempted)
                self.assertFalse(temporary_path.exists())
                self.assertFalse(temporary_path.parent.exists())
            finally:
                cancellation.set()
                thread.join(timeout=3)
                pool.close()

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_preemption_terminates_descendant_processes(self):
        pool = HandlerProcessPool(1, executor=_process_executor)
        cancellation = threading.Event()
        outcome = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "descendant.pid"
            heartbeat = root / "heartbeat"

            def execute_with_descendant():
                try:
                    pool.execute(
                        {
                            "mode": "descendant",
                            "markerPath": str(marker),
                            "heartbeatPath": str(heartbeat),
                        },
                        {},
                        cancellation,
                    )
                except Exception as error:
                    outcome["error"] = error

            thread = threading.Thread(target=execute_with_descendant)
            thread.start()
            try:
                deadline = time.monotonic() + 3
                while (
                    (not marker.exists() or not heartbeat.exists())
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertTrue(marker.exists())
                self.assertTrue(heartbeat.exists())

                cancellation.set()
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive())
                self.assertIsInstance(outcome.get("error"), HandlerPreempted)
                stopped_heartbeat = heartbeat.read_text(encoding="utf-8")
                time.sleep(0.15)
                self.assertEqual(
                    heartbeat.read_text(encoding="utf-8"),
                    stopped_heartbeat,
                )
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
