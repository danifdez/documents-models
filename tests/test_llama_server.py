import io
import json
import os
import signal
import subprocess
import unittest
import urllib.error
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

from services import llama_server

URL = "http://127.0.0.1:18082"


class LlamaEndpointTests(unittest.TestCase):
    def response(self, value):
        return io.BytesIO(json.dumps(value).encode())

    def test_health_checks_return_false_for_an_unavailable_engine(self):
        response = Mock()
        response.getcode.return_value = 200
        with patch.object(
            llama_server.urllib.request, "urlopen", side_effect=[nullcontext(response), TimeoutError]
        ) as request:
            self.assertTrue(llama_server.is_alive(URL))
            self.assertFalse(llama_server.is_alive(URL))
        self.assertEqual(request.call_args_list, [call(URL + "/health", timeout=2.0)] * 2)

    def test_endpoint_readers_route_by_model_and_validate_response_shape(self):
        replies = [
            self.response({"model_path": "/models/base.gguf"}),
            self.response(["wrong"]),
            io.BytesIO(b" " * (1024 * 1024 + 1)),
            self.response([{"id": 0, "path": "/models/adapter.gguf"}]),
            self.response({"wrong": "shape"}),
            io.BytesIO(b" " * (1024 * 1024 + 1)),
            urllib.error.URLError("offline"),
        ]
        with patch.object(llama_server.urllib.request, "urlopen", side_effect=replies) as request:
            self.assertEqual(llama_server.props(URL, model_id="Qwen 3"), {"model_path": "/models/base.gguf"})
            self.assertEqual(llama_server.props(URL), {})
            self.assertEqual(llama_server.props(URL), {})
            self.assertEqual(llama_server.lora_adapters(URL + "/", model_id="Qwen 3"), [
                {"id": 0, "path": "/models/adapter.gguf"},
            ])
            self.assertEqual(llama_server.lora_adapters(URL), [])
            self.assertEqual(llama_server.lora_adapters(URL), [])
            self.assertEqual(llama_server.lora_adapters(URL), [])
        self.assertEqual(request.call_args_list[0], call(URL + "/props?model=Qwen+3", timeout=5.0))
        self.assertEqual(request.call_args_list[3], call(URL + "/lora-adapters?model=Qwen+3", timeout=5.0))

    def test_loaded_model_uses_router_state_then_legacy_props(self):
        router = self.response({"data": [
            {"id": "cold", "status": {"value": "unloaded"}},
            {"id": "hot", "status": {"value": "loaded"}},
        ]})
        with (
            patch.object(llama_server.urllib.request, "urlopen", side_effect=[router, TimeoutError, TimeoutError]),
            patch.object(llama_server, "props", side_effect=[
                {"default_generation_settings": {"model": "/models/legacy.gguf"}}, {},
            ]) as props,
        ):
            self.assertEqual(llama_server.loaded_model(URL), "hot")
            props.assert_not_called()
            self.assertEqual(llama_server.loaded_model(URL), "legacy.gguf")
            self.assertEqual(llama_server.loaded_model(URL), "")


class LlamaConfigurationTests(unittest.TestCase):
    def test_engine_defaults_selects_models_and_respects_environment_overrides(self):
        with (
            patch.dict(os.environ, {
                "LLAMA_SERVER_CTX": "4096", "LLAMA_SERVER_GPU_LAYERS": "all",
                "LLAMA_SERVER_THREADS": "invalid", "LLAMA_SERVER_MODELS_MAX": "0",
            }, clear=True),
            patch.object(llama_server, "get_llm_defaults", return_value={}),
            patch.object(llama_server, "get_tasks", return_value={
                "nested": [{"model": "z.gguf"}, {"llm_model": "a.gguf"}],
                "embedding": {"model": "example/e5"},
            }),
            patch.object(llama_server, "llm_params_for", return_value={
                "model_path": "/models/a.gguf", "n_ctx": 2048, "n_threads": 4, "n_gpu_layers": 20,
            }) as params,
            patch.object(llama_server, "active_deployments", return_value={}),
        ):
            engine = llama_server.engine_defaults()

        params.assert_called_once_with({"model": "a.gguf"})
        self.assertEqual(engine, {
            "model_path": "/models/a.gguf", "models_dir": None, "models_max": 1,
            "n_ctx": 4096, "n_threads": 4, "n_gpu_layers": "all", "lora_paths": [],
        })

    def test_binary_resolution_prefers_explicit_then_executable_bundle_then_path(self):
        with TemporaryDirectory() as directory:
            bundled = Path(directory) / "llama-server"
            bundled.touch(mode=0o700)
            with (
                patch.dict(os.environ, {"LLAMA_SERVER_BIN": "~/custom-engine"}, clear=True),
                patch.object(llama_server, "get_llm_defaults", return_value={}),
                patch.object(llama_server, "_bundled_binaries", return_value=[str(bundled)]),
                patch.object(llama_server.shutil, "which", return_value="/usr/bin/llama-server"),
            ):
                self.assertEqual(llama_server.server_binary(), os.path.expanduser("~/custom-engine"))
                del os.environ["LLAMA_SERVER_BIN"]
                self.assertEqual(llama_server.server_binary(), str(bundled))
                bundled.chmod(0o600)
                self.assertEqual(llama_server.server_binary(), "/usr/bin/llama-server")

    def test_router_command_shares_context_and_registers_adapters_without_applying_them(self):
        with patch.dict(os.environ, {
            "LLAMA_SERVER_SLOTS": "3", "LLAMA_SERVER_ARGS": '--alias "friendly name"',
        }, clear=True):
            command = llama_server.engine_cmd("/engine", URL, {
                "model_path": "/unused.gguf", "models_dir": "/catalog", "models_max": 2,
                "n_ctx": 4096, "n_threads": 4, "n_gpu_layers": "auto", "lora_paths": ["/adapter.gguf"],
            })
        for flag, value in (
            ("--ctx-size", "12288"), ("--port", "18082"), ("--parallel", "3"),
            ("--models-dir", "/catalog"), ("--models-max", "2"), ("--lora", "/adapter.gguf"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--model", command)
        self.assertIn("--lora-init-without-apply", command)
        self.assertEqual(command[-2:], ["--alias", "friendly name"])


class LlamaLifecycleTests(unittest.TestCase):
    def test_ensure_server_reuses_an_existing_endpoint_and_spawns_after_it_dies(self):
        with (
            patch.object(llama_server, "server_url", return_value=URL),
            patch.object(llama_server, "_ready_url", None),
            patch.object(llama_server, "is_alive", side_effect=[True, True, False, False]),
            patch.object(llama_server, "_spawn") as spawn,
        ):
            self.assertEqual(llama_server.ensure_server(), URL)
            self.assertEqual(llama_server.ensure_server(), URL)
            spawn.assert_not_called()
            self.assertEqual(llama_server.ensure_server("/requested.gguf"), URL)
            spawn.assert_called_once_with(URL, "/requested.gguf")

    def test_spawn_requires_a_binary_and_an_existing_model_file(self):
        with (
            patch.object(llama_server, "server_binary", return_value=""),
            self.assertRaisesRegex(RuntimeError, "no llama-server was found"),
        ):
            llama_server._spawn(URL, "")
        with (
            patch.object(llama_server, "server_binary", return_value="/engine"),
            patch.object(llama_server, "engine_defaults", return_value={"model_path": "/missing/model.gguf"}),
            patch.object(llama_server.os.path, "isfile", return_value=False),
            self.assertRaisesRegex(RuntimeError, "Model file not found"),
        ):
            llama_server._spawn(URL, "/missing/also.gguf")

    def test_stop_reaps_a_live_child_and_only_kills_it_after_a_timeout(self):
        for wait_error in (None, OSError("already gone"), subprocess.TimeoutExpired("/engine", 15)):
            with self.subTest(wait_error=wait_error):
                child = Mock(pid=123)
                child.poll.return_value = None
                child.wait.side_effect = wait_error
                with patch.object(llama_server, "_child", child):
                    llama_server.stop()
                    llama_server.stop()
                    self.assertIsNone(llama_server._child)

                child.send_signal.assert_called_once_with(signal.SIGTERM)
                child.wait.assert_called_once_with(timeout=15)
                self.assertEqual(child.kill.call_count, int(isinstance(wait_error, subprocess.TimeoutExpired)))

    def test_cli_can_report_url_and_command_and_detect_missing_configuration(self):
        output = io.StringIO()
        with patch.object(llama_server, "server_url", return_value=URL), redirect_stdout(output):
            self.assertEqual(llama_server.main(["--url"]), 0)
            with (
                patch.object(llama_server, "server_binary", return_value=""),
                self.assertLogs(llama_server.logger, level="ERROR"),
            ):
                self.assertEqual(llama_server.main([]), 2)
            with (
                patch.object(llama_server, "server_binary", return_value="/engine"),
                patch.object(llama_server, "engine_defaults", return_value={"model_path": "/model"}),
                patch.object(llama_server, "engine_cmd", return_value=["/engine", "--model", "/with spaces"]),
                patch.object(llama_server, "is_alive", return_value=True),
                patch.object(llama_server.os.path, "isfile", return_value=True),
                patch.object(llama_server.os, "execve") as execute,
            ):
                self.assertEqual(llama_server.main(["--print-cmd"]), 0)
                self.assertEqual(llama_server.main([]), 0)
                execute.assert_not_called()
        self.assertEqual(output.getvalue().splitlines(), [URL, "/engine --model '/with spaces'"])
