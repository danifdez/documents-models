import io
import json
import os
import signal
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import executions
from lib.execution.protocol_client import (
    ProtocolTransportError,
    WorkerAuthenticationError,
)


class WorkerSelfCheckTests(unittest.TestCase):
    def check(self, binary, metadata, handler_available=True):
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"DOCUMENTS_MODELS_VARIANT": "cpu"}),
            patch.object(executions, "CAPABILITIES", ["sample"]),
            patch("utils.task_dispatch.ensure_task_handler", return_value=handler_available),
            patch("services.llama_server.server_binary", return_value=binary),
            patch("utils.runtime_variant.llama_build_metadata", side_effect=[metadata]) as build,
            redirect_stdout(output),
        ):
            status = executions.self_check()

        if binary:
            build.assert_called_once_with(binary)
        else:
            build.assert_not_called()
        return status, json.loads(output.getvalue())

    def test_healthy_bundle_reports_its_resources_and_compatible_engine(self):
        status, result = self.check("/example/llama-server", "ggml CPU")
        self.assertEqual(status, 0)
        self.assertTrue(result["ok"])
        self.assertEqual(result["failedHandlers"], [])
        self.assertEqual(result["resources"], {"defaults": True, "prompts": True})
        self.assertEqual(result["llamaServer"], {
            "available": True, "compatible": True, "path": "/example/llama-server",
            "build": "ggml CPU", "error": "",
        })

    def test_reports_missing_handlers_and_missing_or_broken_engine(self):
        for binary, metadata, reason in (
            ("", "", "llama-server was not found"),
            ("/example/llama-server", OSError("cannot run engine"), "cannot run engine"),
            ("/example/llama-server", "ggml_cuda", "CUDA-enabled"),
        ):
            with self.subTest(binary=binary, metadata=metadata):
                status, result = self.check(binary, metadata, handler_available=False)
                self.assertEqual(status, 1)
                self.assertEqual(result["failedHandlers"], ["sample"])
                self.assertFalse(result["llamaServer"]["compatible"])
                self.assertIn(reason, result["llamaServer"]["error"])

    def test_capabilities_keep_catalog_order_and_only_include_supported_tasks(self):
        with (
            patch.object(executions, "detect_worker_capabilities", return_value={"cpu": True}),
            patch.object(executions, "get_supported_task_types", return_value=["second", "first"]) as supported,
            patch.object(executions, "get_all_task_names", return_value=["first", "disabled", "second"]),
        ):
            self.assertEqual(executions.effective_task_capabilities(), ["first", "second"])
        supported.assert_called_once_with({"cpu": True})


class WorkerLoopTests(unittest.TestCase):
    def test_reenrolls_retries_and_closes_after_a_signal(self):
        signal_handlers = {}
        assignment = {"attemptId": "example"}
        runtime = Mock(available_slots=1, active_attempt_ids={"previous"})
        runtime.seconds_until_maintenance.return_value = 2
        client = Mock()
        client.ensure_registered.side_effect = [None, WorkerAuthenticationError("expired"), None, None, None]
        client.claim.side_effect = [ProtocolTransportError("offline"), assignment, None]
        metadata = {"codeFingerprint": "code", "runtimeFingerprint": "runtime"}
        outbox = Mock()

        def retry_or_stop(_delay):
            if client.claim.call_count == 3:
                signal_handlers[signal.SIGTERM](signal.SIGTERM, None)

        with (
            patch.object(executions, "ExecutionProtocolClient", return_value=client),
            patch.object(executions, "WorkerRuntime", return_value=runtime) as make_runtime,
            patch.object(executions, "ResultOutbox", return_value=outbox) as make_outbox,
            patch.object(executions, "worker_data_dir", return_value="/example/outbox"),
            patch.object(executions, "log_hardware_summary"),
            patch.object(executions, "_metadata", return_value=metadata),
            patch.object(executions, "CAPABILITIES", ["sample"]),
            patch.object(executions, "signal", SimpleNamespace(
                SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, signal=signal_handlers.__setitem__,
            )),
            patch.object(executions, "time", SimpleNamespace(monotonic=lambda: 10, sleep=retry_or_stop)),
            self.assertLogs(executions.logger, level="WARNING") as logs,
        ):
            executions.main()

        make_outbox.assert_called_once_with("/example/outbox")
        make_runtime.assert_called_once_with(client, outbox, executions.MAXIMUM_CONCURRENCY, 30_000)
        client.reset_credential.assert_called_once_with()
        client.heartbeat.assert_called_once_with(["sample"], executions.STEP_KINDS, executions.MAXIMUM_CONCURRENCY, metadata)
        self.assertEqual(client.claim.call_args_list, [call(["sample"], executions.STEP_KINDS, 30_000, 1_000)] * 3)
        runtime.submit.assert_called_once_with(assignment)
        runtime.close.assert_called_once_with()
        self.assertEqual(set(signal_handlers), {signal.SIGINT, signal.SIGTERM})
        self.assertTrue(any("re-enrolling" in message for message in logs.output))
        self.assertTrue(any("offline" in message for message in logs.output))

    def test_defers_claiming_until_a_slot_is_available_and_cleans_up_on_failure(self):
        runtime = Mock(available_slots=0, active_attempt_ids={"current"})
        runtime.seconds_until_maintenance.return_value = 0.5
        runtime.wait_for_completion.side_effect = RuntimeError("worker failure")
        client = Mock()
        with (
            patch.object(executions, "ExecutionProtocolClient", return_value=client),
            patch.object(executions, "WorkerRuntime", return_value=runtime),
            patch.object(executions, "ResultOutbox"),
            patch.object(executions, "worker_data_dir"),
            patch.object(executions, "log_hardware_summary"),
            patch.object(executions, "signal", Mock(SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM)),
            patch.object(executions, "time", SimpleNamespace(monotonic=lambda: 10)),
            self.assertRaisesRegex(RuntimeError, "worker failure"),
        ):
            executions.main()

        client.claim.assert_not_called()
        runtime.maintain_leases.assert_called_once_with(10)
        runtime.wait_for_completion.assert_called_once_with(0.5)
        runtime.close.assert_called_once_with()
