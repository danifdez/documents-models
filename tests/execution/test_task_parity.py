import copy
import os
import unittest
from unittest.mock import patch

from agents.loop import run_agent_loop
from lib.framework.agent import AgentSpec
from lib.framework.tool import ToolContext
from lib.execution.emitter import ExecutionEmitter
from tasks.assistant_chat.assistant_chat import assistant_chat

from tests.execution.test_emitter import CONTEXT


class AssistantChatExecutionTest(unittest.TestCase):
    def run_handler(self, reply="stable reply", failure=None):
        payload = {
            "ownerId": 1,
            "assistantSystem": False,
            "conversation": [{"role": "user", "content": "hello"}],
            "execution": copy.deepcopy(CONTEXT),
        }
        environment = {
            "EXECUTION_INGEST_TOKEN": "test-token",
        }
        generate = failure if failure else lambda *_args, **_kwargs: reply
        if failure:
            generate = unittest.mock.Mock(side_effect=failure)

        with patch.dict(os.environ, environment, clear=False), patch(
            "tasks.assistant_chat.assistant_chat.get_task_config",
            return_value={"max_tokens": 32, "stream": False},
        ), patch(
            "tasks.assistant_chat.assistant_chat.get_llm_params", return_value={}
        ), patch(
            "tasks.assistant_chat.assistant_chat.get_llm_service", return_value=object()
        ), patch(
            "tasks.assistant_chat.assistant_chat.build_chat_messages",
            return_value=[{"role": "user", "content": "hello"}],
        ), patch(
            "tasks.assistant_chat.assistant_chat.generate_reply", generate
        ), patch(
            "lib.execution.emitter.ExecutionEmitter._post",
            return_value={"accepted": 100, "duplicates": 0},
        ):
            return assistant_chat(payload)

    def product_result(self, value):
        result = dict(value)
        result.pop("executionTelemetry", None)
        return result

    def test_returns_reply_with_execution_telemetry(self):
        result = self.run_handler()
        self.assertEqual(result["reply"], "stable reply")
        self.assertGreater(result["executionTelemetry"]["attemptedEvents"], 0)

    def test_inference_failure_keeps_the_same_product_error(self):
        result = self.run_handler(failure=RuntimeError("inference unavailable"))
        self.assertEqual(result["error"], "Assistant failure: inference unavailable")


class ToolRunTest(unittest.TestCase):
    def emitter(self):
        environment = {
            "EXECUTION_INGEST_TOKEN": "test-token",
        }
        with patch.dict(os.environ, environment, clear=False):
            return ExecutionEmitter(copy.deepcopy(CONTEXT))

    def run_loop(self, dispatch):
        class Llm:
            def __init__(self):
                self.round = 0

            def chat_with_tools(self, *_args, **_kwargs):
                self.round += 1
                if self.round == 1:
                    return {
                        "content": "",
                        "tool_calls": [{
                            "id": "provider-call-1",
                            "function": {
                                "name": "read_fixture",
                                "arguments": '{"query":"item"}',
                            },
                        }],
                    }
                return {"content": "done", "tool_calls": []}

        emitter = self.emitter()
        spec = AgentSpec(
            name="test-agent",
            config_key="test-agent",
            system_prompt="test",
            tool_names=frozenset({"read_fixture"}),
        )
        ctx = ToolContext(execution=emitter)
        with patch("agents.loop.get_task_config", return_value={"max_rounds": 2}), patch(
            "agents.loop.get_llm_params", return_value={}
        ), patch("agents.loop.get_llm_service", return_value=Llm()):
            try:
                result = run_agent_loop(spec, [], ctx, [], dispatch)
                return emitter, result, None
            except Exception as error:  # noqa: BLE001
                return emitter, None, error

    def test_read_tool_has_correlated_start_source_and_finish(self):
        emitter, _result, error = self.run_loop(
            lambda *_: {"summary": "fixture value"})

        self.assertIsNone(error)
        operation = [event for event in emitter.pending_events
                     if event["eventType"].startswith("operation.")]
        source = next(event for event in emitter.pending_events
                      if event["eventType"] == "source.observed")
        self.assertEqual([event["eventType"] for event in operation], [
            "operation.started", "operation.finished",
        ])
        self.assertEqual(operation[0]["operationId"], operation[1]["operationId"])
        self.assertEqual(source["operationId"], operation[0]["operationId"])
        self.assertEqual(operation[1]["causedByEventId"], source["eventId"])
        self.assertEqual(operation[1]["payload"]["status"], "succeeded")

    def test_tool_exception_closes_the_operation_as_failed(self):
        def fail(*_args):
            raise RuntimeError("tool unavailable")

        emitter, _result, error = self.run_loop(fail)

        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "tool unavailable")
        finish = next(event for event in emitter.pending_events
                      if event["eventType"] == "operation.finished")
        self.assertEqual(finish["payload"]["status"], "failed")
        self.assertEqual(finish["payload"]["error"]["message"], "tool unavailable")


if __name__ == "__main__":
    unittest.main()
