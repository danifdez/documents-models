import os
import unittest
from unittest.mock import Mock, mock_open, patch

from services import llama_server


class EngineEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "Authorization=PRIVATE",
            "OTEL_EXPORTER_OTLP_CLIENT_KEY": "/private/client.key",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example",
            "ENGINE_TEST_SETTING": "preserved",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def assert_private_environment(self, environment):
        self.assertFalse(any(key.startswith("OTEL_EXPORTER_OTLP") for key in environment))
        self.assertEqual(environment["ENGINE_TEST_SETTING"], "preserved")
        self.assertEqual(os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"], "Authorization=PRIVATE")

    def test_foreground_engine_never_inherits_exporter_credentials(self):
        with patch.object(llama_server, "server_binary", return_value="/engine"), \
             patch.object(llama_server, "engine_defaults", return_value={"model_path": "/model"}), \
             patch.object(llama_server, "engine_cmd", return_value=["/engine"]), \
             patch.object(llama_server.os.path, "isfile", return_value=True), \
             patch.object(llama_server, "is_alive", return_value=False), \
             patch.object(llama_server.os, "execve") as execute:
            llama_server.main([])
        self.assert_private_environment(execute.call_args.args[2])

    def test_fallback_engine_never_inherits_exporter_credentials(self):
        child = Mock()
        child.poll.return_value = None
        with patch.object(llama_server, "server_binary", return_value="/engine"), \
             patch.object(llama_server, "engine_defaults", return_value={"model_path": "/model"}), \
             patch.object(llama_server, "engine_cmd", return_value=["/engine"]), \
             patch.object(llama_server.os.path, "isfile", return_value=True), \
             patch.object(llama_server, "is_alive", return_value=True), \
             patch("builtins.open", mock_open()), \
             patch.object(llama_server.atexit, "register"), \
             patch.object(llama_server, "_child", None), \
             patch.object(llama_server.subprocess, "Popen", return_value=child) as spawn:
            llama_server._spawn("http://127.0.0.1:18082", "/model")
        self.assert_private_environment(spawn.call_args.kwargs["env"])
