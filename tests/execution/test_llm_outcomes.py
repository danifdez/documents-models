import base64
import copy
import json
import os
import unittest
from unittest.mock import Mock, patch

from lib.execution import activate_emitter, reset_emitter
from agents.memory_agent import extract_memory_action
from services.llm_service import LLMService

from tests.execution.test_emitter import CONTEXT
from tests.execution.support import RecordingExecutionEmitter


class LlmOutcomeTraceTest(unittest.TestCase):
    def llm(self):
        service = object.__new__(LLMService)
        service.url = "http://llama.test"
        service.sampling = {}
        service._lora_id = None
        return service

    def emitter(self):
        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": "test-token"}, clear=False):
            return RecordingExecutionEmitter(copy.deepcopy(CONTEXT))

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
            event for event in emitter.sent_events
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
            event for event in emitter.sent_events
            if event["eventType"] == "operation.finished"
        )
        self.assertEqual(finish["payload"]["outcome"], "tool_requests")

    def test_output_repair_records_its_limit_and_outcome(self):
        emitter = self.emitter()
        token = activate_emitter(emitter)
        empty_response = {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "",
                    "reasoning_content": "private reasoning",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }
        repaired_response = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "repaired answer"},
            }],
            "usage": {"prompt_tokens": 14, "completion_tokens": 3},
        }
        try:
            with patch(
                "services.llm_service._post",
                side_effect=[empty_response, repaired_response],
            ):
                invalid = self.llm().chat_with_tools(
                    [{"role": "user", "content": "question"}],
                    [{"type": "function", "function": {"name": "read_fixture"}}],
                )
                result = self.llm().chat_with_tools(
                    [{"role": "user", "content": "question"}],
                    [{"type": "function", "function": {"name": "read_fixture"}}],
                    inference_name="output_repair",
                    trace_metadata={
                        "phase": "output_repair",
                        "reason": "empty_model_response",
                        "attempt": 1,
                        "maxAttempts": 1,
                    },
                )
        finally:
            reset_emitter(token)

        self.assertEqual(invalid["content"], "")
        self.assertEqual(result["content"], "repaired answer")
        invalid_start, invalid_finish, repair_start, repair_finish = [
            event for event in emitter.sent_events
            if event["eventType"].startswith("operation.")
        ]
        self.assertEqual(invalid_finish["payload"]["outcome"], "invalid")
        self.assertEqual(
            invalid_finish["payload"]["reason"], "empty_model_response"
        )
        self.assertEqual(repair_start["causedByEventId"], invalid_finish["eventId"])
        self.assertEqual(repair_start["payload"]["name"], "output_repair")
        self.assertEqual(repair_start["payload"]["phase"], "output_repair")
        self.assertEqual(repair_start["payload"]["reason"], "empty_model_response")
        self.assertEqual(repair_start["payload"]["attempt"], 1)
        self.assertEqual(repair_start["payload"]["maxAttempts"], 1)
        self.assertEqual(repair_finish["payload"]["outcome"], "final_text")

        invalid_result = invalid_finish["payload"]["result"]
        self.assertEqual(
            invalid_finish["artifactRefs"],
            [
                invalid_result["responseArtifactId"],
                invalid_result["rawResponseArtifactId"],
            ],
        )
        artifacts = {
            artifact["artifactId"]: json.loads(
                base64.b64decode(artifact["bodyBase64"])
            )
            for artifact in emitter.sent_artifacts
        }
        self.assertEqual(
            artifacts[invalid_result["responseArtifactId"]],
            {"content": "", "tool_calls": []},
        )
        raw_response = artifacts[invalid_result["rawResponseArtifactId"]]
        self.assertEqual(raw_response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(raw_response["usage"]["completion_tokens"], 1)
        self.assertNotIn("reasoning_content", raw_response["choices"][0]["message"])
        repair_result = repair_finish["payload"]["result"]
        self.assertEqual(
            artifacts[repair_result["responseArtifactId"]],
            {"content": "repaired answer", "tool_calls": []},
        )
        repair_raw_response = artifacts[repair_result["rawResponseArtifactId"]]
        self.assertEqual(
            repair_raw_response["usage"]["completion_tokens"],
            3,
        )

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
            event for event in emitter.sent_events
            if event["eventType"] == "operation.started"
        )
        finish = next(
            event for event in emitter.sent_events
            if event["eventType"] == "operation.finished"
        )
        self.assertEqual(start["payload"]["name"], "memory_extraction")
        self.assertEqual(start["payload"]["phase"], "memory_extraction")
        self.assertEqual(finish["payload"]["outcome"], "final_text")

    def test_required_start_evidence_fails_before_model_dispatch(self):
        emitter = self.emitter()
        emitter.recording_client.responder = lambda *_args, **_kwargs: None
        token = activate_emitter(emitter)
        model_post = Mock()
        try:
            with patch("services.llm_service._post", model_post):
                with self.assertRaisesRegex(RuntimeError, "artifact ingestion"):
                    self.llm().chat(
                        [{"role": "user", "content": "question"}],
                        inference_name="direct_response",
                        trace_metadata={"phase": "direct_response"},
                    )
        finally:
            reset_emitter(token)

        model_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
