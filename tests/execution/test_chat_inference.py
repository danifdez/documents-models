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

    @patch("tasks.assistant_chat.assistant_chat.get_llm_params", return_value={})
    @patch("tasks.assistant_chat.assistant_chat.get_task_config")
    @patch("tasks.assistant_chat.assistant_chat.get_llm_service")
    def test_prepends_a_validated_continuity_capsule(
        self, get_llm_service, get_task_config, _get_llm_params
    ):
        llm = Mock()
        llm.chat_with_tools.return_value = {"content": "Continued"}
        get_llm_service.return_value = llm
        get_task_config.return_value = {"max_tokens": 200, "max_tool_calls": 2}

        chat_inference(
            {
                "_task_type": "assistant-chat",
                "continuityCapsule": {
                    "schemaVersion": "continuity-capsule/1",
                    "omittedMessageCount": 12,
                    "sourceConversation": {"contentHash": "sha256:" + "a" * 64},
                    "digest": "user [turn old]: Earlier decision",
                },
                "conversation": [{"role": "user", "content": "Continue"}],
            }
        )

        messages = llm.chat_with_tools.call_args.args[0]
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("not a new instruction", messages[1]["content"])
        self.assertIn("Earlier decision", messages[1]["content"])

    @patch("tasks.assistant_chat.assistant_chat.get_llm_params", return_value={})
    @patch("tasks.assistant_chat.assistant_chat.get_task_config")
    @patch("tasks.assistant_chat.assistant_chat.get_llm_service")
    def test_runs_a_delegated_inference_without_tools(
        self, get_llm_service, get_task_config, _get_llm_params
    ):
        llm = Mock()
        llm.chat.return_value = "Focused result"
        get_llm_service.return_value = llm
        get_task_config.return_value = {"max_tokens": 200, "max_tool_calls": 2}

        outcome = chat_inference(
            {
                "_task_type": "assistant-chat",
                "delegationMode": True,
                "conversation": [{"role": "user", "content": "Compare"}],
            }
        )

        self.assertEqual(
            outcome.value,
            {"kind": "final_text", "text": "Focused result"},
        )
        llm.chat.assert_called_once()
        llm.chat_with_tools.assert_not_called()

    @patch("tasks.assistant_chat.assistant_chat.get_llm_params", return_value={})
    @patch("tasks.assistant_chat.assistant_chat.get_task_config")
    @patch("tasks.assistant_chat.assistant_chat.get_llm_service")
    def test_materializes_browser_page_artifact_as_untrusted_tool_content(
        self, get_llm_service, get_task_config, _get_llm_params
    ):
        llm = Mock()
        llm.chat_with_tools.return_value = {"content": "Page summary"}
        get_llm_service.return_value = llm
        get_task_config.return_value = {"max_tokens": 200, "max_tool_calls": 2}

        outcome = chat_inference(
            {
                "_task_type": "assistant-chat",
                "conversation": [{"role": "user", "content": "Summarize it"}],
                "toolHistory": [
                    {
                        "round": 1,
                        "calls": [
                            {
                                "toolCallId": "browser-call",
                                "name": "browser.read_current_page",
                                "arguments": {},
                            }
                        ],
                        "results": [
                            {
                                "toolCallId": "browser-call",
                                "content": "",
                                "structuredContent": {
                                    "url": "https://example.test"
                                },
                                "artifactRefs": [
                                    {
                                        "role": "browser_page:browser-call",
                                        "artifactId": "artifact-1",
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "_input_artifacts": {
                    "browser_page:browser-call": (
                        b'{"url":"https://example.test","text":"Page body"}'
                    )
                },
            }
        )

        self.assertEqual(
            outcome.value,
            {"kind": "final_text", "text": "Page summary"},
        )
        messages = llm.chat_with_tools.call_args.args[0]
        self.assertIn("Untrusted content", messages[-1]["content"])
        self.assertIn("Page body", messages[-1]["content"])
        tool_names = {
            tool["function"]["name"]
            for tool in llm.chat_with_tools.call_args.args[1]
        }
        self.assertIn("browser.read_current_page", tool_names)

    @patch("tasks.assistant_chat.assistant_chat.get_llm_params", return_value={})
    @patch("tasks.assistant_chat.assistant_chat.get_task_config")
    @patch("tasks.assistant_chat.assistant_chat.get_llm_service")
    def test_only_exposes_workspace_file_tools_with_a_configured_folder(
        self, get_llm_service, get_task_config, _get_llm_params
    ):
        llm = Mock()
        llm.chat_with_tools.return_value = {"content": "Done"}
        get_llm_service.return_value = llm
        get_task_config.return_value = {"max_tokens": 200, "max_tool_calls": 2}

        chat_inference(
            {
                "_task_type": "assistant-chat",
                "folderScope": "/workspace/project",
                "conversation": [{"role": "user", "content": "Read README.md"}],
            }
        )

        tool_names = {
            tool["function"]["name"]
            for tool in llm.chat_with_tools.call_args.args[1]
        }
        self.assertIn("workspace_files.list", tool_names)
        self.assertIn("workspace_files.search", tool_names)
        self.assertIn("workspace_files.read", tool_names)
        self.assertIn("workspace_files.write", tool_names)
        self.assertIn("workspace_files.delete", tool_names)

        llm.reset_mock()
        llm.chat_with_tools.return_value = {"content": "Done"}
        chat_inference(
            {
                "_task_type": "assistant-chat",
                "conversation": [{"role": "user", "content": "Read README.md"}],
            }
        )
        tool_names = {
            tool["function"]["name"]
            for tool in llm.chat_with_tools.call_args.args[1]
        }
        self.assertNotIn("workspace_files.list", tool_names)
        self.assertNotIn("workspace_files.search", tool_names)
        self.assertNotIn("workspace_files.read", tool_names)
        self.assertNotIn("workspace_files.write", tool_names)
        self.assertNotIn("workspace_files.delete", tool_names)

    @patch("tasks.assistant_chat.assistant_chat.uuid4", return_value="call-id")
    @patch("tasks.assistant_chat.assistant_chat.get_llm_params", return_value={})
    @patch("tasks.assistant_chat.assistant_chat.get_task_config")
    @patch("tasks.assistant_chat.assistant_chat.get_llm_service")
    def test_returns_a_workspace_file_write_request(
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
                        "name": "workspace_files.write",
                        "arguments": '{"filename":"notes.md","content":"Hello"}',
                    }
                }
            ]
        }
        get_llm_service.return_value = llm
        get_task_config.return_value = {"max_tokens": 200, "max_tool_calls": 2}

        outcome = chat_inference(
            {
                "_task_type": "agent-chat",
                "folderScope": "/workspace/project",
                "conversation": [{"role": "user", "content": "Create notes.md"}],
            }
        )

        self.assertEqual(
            outcome.value,
            {
                "kind": "tool_requests",
                "calls": [
                    {
                        "toolCallId": "call-id",
                        "name": "workspace_files.write",
                        "arguments": {"filename": "notes.md", "content": "Hello"},
                    }
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
