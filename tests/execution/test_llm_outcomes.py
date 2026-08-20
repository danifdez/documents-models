import copy
import os
import unittest
from unittest.mock import patch

from lib.execution import activate_emitter, reset_emitter
from lib.execution.emitter import ExecutionEmitter
from agents.memory_agent import extract_memory_action
from services.llm_service import LLMService

from tests.execution.test_emitter import CONTEXT


class LlmOutcomeTraceTest(unittest.TestCase):
    def llm(self):
        service = object.__new__(LLMService)
        service.url = "http://llama.test"
        service.sampling = {}
        service._lora_id = None
        return service

    def emitter(self):
        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": "test-token"}, clear=False):
            return ExecutionEmitter(copy.deepcopy(CONTEXT))

    def test_forced_finalization_records_its_reason_and_final_text_outcome(self):
        emitter = self.emitter()
        token = activate_emitter(emitter)
        response = {
            "choices": [{"message": {"content": "budget answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }
        try:
            with patch("services.llm_service._post", return_value=response):
                result = self.llm().chat(
                    [{"role": "user", "content": "question"}],
                    inference_name="forced_finalization",
                    trace_metadata={
                        "phase": "forced_finalization",
                        "reason": "step_budget_exhausted",
                    },
                )
        finally:
            reset_emitter(token)

        self.assertEqual(result, "budget answer")
        start, finish = [
            event for event in emitter.pending_events
            if event["eventType"].startswith("operation.")
        ]
        self.assertEqual(start["payload"]["name"], "forced_finalization")
        self.assertEqual(start["payload"]["phase"], "forced_finalization")
        self.assertEqual(start["payload"]["reason"], "step_budget_exhausted")
        self.assertEqual(finish["payload"]["outcome"], "final_text")

    def test_tool_request_has_one_normalized_outcome(self):
        emitter = self.emitter()
        token = activate_emitter(emitter)
        response = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {"name": "read_fixture", "arguments": "{}"},
                    }],
                },
            }],
        }
        try:
            with patch("services.llm_service._post", return_value=response):
                self.llm().chat_with_tools(
                    [{"role": "user", "content": "question"}],
                    [{"type": "function", "function": {"name": "read_fixture"}}],
                )
        finally:
            reset_emitter(token)

        finish = next(
            event for event in emitter.pending_events
            if event["eventType"] == "operation.finished"
        )
        self.assertEqual(finish["payload"]["outcome"], "tool_requests")

    def test_memory_extraction_is_distinguishable_from_the_user_reply(self):
        emitter = self.emitter()
        token = activate_emitter(emitter)
        response = {
            "choices": [{"message": {"content": '{"action":"none"}'}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }
        try:
            with patch("services.llm_service._post", return_value=response):
                result = extract_memory_action(
                    self.llm(),
                    "hello",
                    [],
                    {"memory_extract_max_tokens": 32},
                )
        finally:
            reset_emitter(token)

        self.assertIsNone(result)
        start = next(
            event for event in emitter.pending_events
            if event["eventType"] == "operation.started"
        )
        finish = next(
            event for event in emitter.pending_events
            if event["eventType"] == "operation.finished"
        )
        self.assertEqual(start["payload"]["name"], "memory_extraction")
        self.assertEqual(start["payload"]["phase"], "memory_extraction")
        self.assertEqual(finish["payload"]["outcome"], "final_text")


if __name__ == "__main__":
    unittest.main()
