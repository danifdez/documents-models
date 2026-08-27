import json
import unittest
from unittest.mock import Mock, patch

from tasks.browser_inference.browser_inference import browser_inference
from lib.execution.step_executor import execute_assignment


class BrowserInferenceTest(unittest.TestCase):
    @patch("tasks.browser_inference.browser_inference.get_llm_service")
    @patch("tasks.browser_inference.browser_inference.get_llm_params")
    def test_executes_as_a_canonical_inference_step(
        self, get_params: Mock, get_service: Mock
    ) -> None:
        get_params.return_value = {}
        get_service.return_value.chat.return_value = "Backend-owned response"
        result = execute_assignment(
            {
                "executionId": "execution-1",
                "stepId": "step-1",
                "operationId": "operation-1",
                "attemptId": "attempt-1",
                "stepKind": "inference",
                "work": {
                    "taskType": "browser-inference",
                    "payload": {
                        "requestJson": json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": "Hello"}
                                ]
                            }
                        )
                    },
                },
            }
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            result["output"]["outcome"]["value"],
            {"content": "Backend-owned response"},
        )

    @patch("tasks.browser_inference.browser_inference.get_llm_service")
    @patch("tasks.browser_inference.browser_inference.get_llm_params")
    def test_preserves_structured_request_and_sampling(
        self, get_params: Mock, get_service: Mock
    ) -> None:
        get_params.return_value = {}
        llm = get_service.return_value
        llm.chat.return_value = '{"decision":"continue"}'
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "decision": {"type": "string"},
            },
        }
        request = {
            "messages": [{"role": "user", "content": "Decide"}],
            "max_tokens": 256,
            "temperature": 0.0,
            "top_k": 1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }

        result = browser_inference(
            {"requestJson": json.dumps(request), "label": "agent-step"}
        )

        self.assertEqual(result, {"content": '{"decision":"continue"}'})
        llm.chat.assert_called_once_with(
            request["messages"],
            max_tokens=256,
            response_format=request["response_format"],
            allow_thinking=True,
            inference_name="agent-step",
            sampling_overrides={"temperature": 0.0, "top_k": 1},
            chat_template_kwargs={"enable_thinking": False},
        )

    def test_rejects_non_text_messages(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported role"):
            browser_inference(
                {
                    "requestJson": json.dumps(
                        {"messages": [{"role": "tool", "content": "x"}]}
                    )
                }
            )


if __name__ == "__main__":
    unittest.main()
