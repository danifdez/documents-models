import unittest
from unittest.mock import Mock, patch

from tasks.assistant_chat.assistant_chat import _SYSTEM_PROMPT, chat_inference


class ChatInferenceTest(unittest.TestCase):
    def test_loads_the_packaged_system_prompt(self):
        self.assertIn("exactly one inference", _SYSTEM_PROMPT)

    @patch("tasks.assistant_chat.assistant_chat.get_llm_params", return_value={})
    @patch("tasks.assistant_chat.assistant_chat.get_task_config")
    @patch("tasks.assistant_chat.assistant_chat.get_llm_service")
    def test_returns_final_text_from_one_inference(
        self, get_llm_service, get_task_config, _get_llm_params
    ):
        llm = Mock()
        llm.chat_with_tools.return_value = {"content": "Respuesta final"}
        get_llm_service.return_value = llm
        get_task_config.return_value = {"max_tokens": 200, "max_tool_calls": 2}

        outcome = chat_inference(
            {
                "_task_type": "assistant-chat",
                "conversation": [{"role": "user", "content": "Responde"}],
            }
        )

        self.assertEqual(
            outcome.value,
            {"kind": "final_text", "text": "Respuesta final"},
        )
        llm.chat_with_tools.assert_called_once()

    @patch("tasks.assistant_chat.assistant_chat.uuid4", return_value="call-id")
    @patch("tasks.assistant_chat.assistant_chat.get_llm_params", return_value={})
    @patch("tasks.assistant_chat.assistant_chat.get_task_config")
    @patch("tasks.assistant_chat.assistant_chat.get_llm_service")
    def test_returns_tool_requests_and_replays_durable_tool_results(
        self,
        get_llm_service,
        get_task_config,
        _get_llm_params,
        _uuid4,
    ):
        llm = Mock()
        llm.chat_with_tools.return_value = {
            "tool_calls": [
                {
                    "function": {
                        "name": "documents.search",
                        "arguments": '{"query":"harness"}',
                    }
                }
            ]
        }
        get_llm_service.return_value = llm
        get_task_config.return_value = {"max_tokens": 200, "max_tool_calls": 2}

        outcome = chat_inference(
            {
                "_task_type": "agent-chat",
                "conversation": [{"role": "user", "content": "Busca"}],
                "toolHistory": [
                    {
                        "round": 1,
                        "calls": [
                            {
                                "toolCallId": "previous-call",
                                "name": "documents.search",
                                "arguments": {"query": "plan"},
                            }
                        ],
                        "results": [
                            {
                                "toolCallId": "previous-call",
                                "content": "Plan result",
                            }
                        ],
                    }
                ],
            }
        )

        self.assertEqual(
            outcome.value,
            {
                "kind": "tool_requests",
                "calls": [
                    {
                        "toolCallId": "call-id",
                        "name": "documents.search",
                        "arguments": {"query": "harness"},
                    }
                ],
            },
        )
        messages = llm.chat_with_tools.call_args.args[0]
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["content"], "Plan result")


if __name__ == "__main__":
    unittest.main()
