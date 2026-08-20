import unittest
from unittest.mock import patch

from agents.loop import run_agent_loop
from lib.framework.agent import AgentRunResult, AgentSpec
from lib.framework.tool import ToolContext


class FakeLlm:
    def __init__(self, messages, forced_reply=""):
        self.messages = list(messages)
        self.forced_reply = forced_reply
        self.tool_calls = []
        self.chat_calls = []

    def chat_with_tools(self, *_args, **_kwargs):
        self.tool_calls.append((_args, _kwargs))
        return self.messages.pop(0)

    def chat(self, *_args, **_kwargs):
        self.chat_calls.append((_args, _kwargs))
        return self.forced_reply


class AgentLoopResultTest(unittest.TestCase):
    def run_loop(self, llm, dispatch=lambda *_args: {"value": "fixture"}, max_rounds=3):
        spec = AgentSpec(
            name="test-agent",
            config_key="test-agent",
            system_prompt="test",
            tool_names=frozenset({"read_fixture"}),
        )
        with patch(
            "agents.loop.get_task_config",
            return_value={"max_rounds": max_rounds, "max_tokens": 32},
        ), patch("agents.loop.get_llm_params", return_value={}), patch(
            "agents.loop.get_llm_service", return_value=llm
        ):
            return run_agent_loop(
                spec,
                [{"role": "user", "content": "question"}],
                ToolContext(),
                [{"type": "function", "function": {"name": "read_fixture"}}],
                dispatch,
            )

    def test_returns_direct_final_text_from_the_same_inference(self):
        llm = FakeLlm([{"content": "direct answer", "tool_calls": []}])

        result = self.run_loop(llm)

        self.assertEqual(result, AgentRunResult.final_text("direct answer"))
        self.assertEqual(len(llm.tool_calls), 1)
        self.assertEqual(llm.chat_calls, [])

    def test_returns_final_text_after_a_native_tool_call(self):
        llm = FakeLlm([
            {
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "function": {
                        "name": "read_fixture",
                        "arguments": '{"query":"item"}',
                    },
                }],
            },
            {"content": "answer from fixture", "tool_calls": []},
        ])
        dispatched = []

        result = self.run_loop(
            llm,
            dispatch=lambda name, arguments, _ctx: (
                dispatched.append((name, arguments)) or {"value": "fixture"}
            ),
        )

        self.assertEqual(result, AgentRunResult.final_text("answer from fixture"))
        self.assertEqual(dispatched, [("read_fixture", '{"query":"item"}')])
        self.assertEqual(len(llm.tool_calls), 2)
        self.assertEqual(llm.chat_calls, [])

    def test_returns_final_text_after_an_inline_tool_call(self):
        llm = FakeLlm([
            {
                "content": (
                    '<tool_call>{"name":"read_fixture",'
                    '"arguments":{"query":"item"}}</tool_call>'
                ),
                "tool_calls": [],
            },
            {"content": "inline answer", "tool_calls": []},
        ])

        result = self.run_loop(llm)

        self.assertEqual(result, AgentRunResult.final_text("inline answer"))
        self.assertEqual(len(llm.tool_calls), 2)
        self.assertEqual(llm.chat_calls, [])

    def test_preserves_the_order_of_multiple_tool_results(self):
        llm = FakeLlm([
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "first", "arguments": '{"index":1}'},
                    },
                    {
                        "id": "call-2",
                        "function": {"name": "second", "arguments": '{"index":2}'},
                    },
                ],
            },
            {"content": "combined answer", "tool_calls": []},
        ])
        dispatched = []

        result = self.run_loop(
            llm,
            dispatch=lambda name, arguments, _ctx: (
                dispatched.append((name, arguments)) or {"value": name}
            ),
        )

        self.assertEqual(result, AgentRunResult.final_text("combined answer"))
        self.assertEqual(dispatched, [
            ("first", '{"index":1}'),
            ("second", '{"index":2}'),
        ])

    def test_empty_output_is_an_explicit_invalid_result(self):
        llm = FakeLlm([{"content": "", "tool_calls": []}])

        result = self.run_loop(llm)

        self.assertEqual(result, AgentRunResult.invalid("empty_model_response"))
        self.assertEqual(llm.chat_calls, [])

    def test_round_exhaustion_uses_one_explicit_forced_finalization(self):
        tool_request = {
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "read_fixture", "arguments": "{}"},
            }],
        }
        llm = FakeLlm([tool_request], forced_reply="budget answer")

        result = self.run_loop(llm, max_rounds=1)

        self.assertEqual(result, AgentRunResult.final_text("budget answer"))
        self.assertEqual(len(llm.chat_calls), 1)
        self.assertEqual(llm.chat_calls[0][1]["inference_name"], "forced_finalization")
        self.assertEqual(
            llm.chat_calls[0][1]["trace_metadata"],
            {"phase": "forced_finalization", "reason": "step_budget_exhausted"},
        )

    def test_structured_agents_keep_their_structured_result(self):
        llm = FakeLlm([{"content": '{"summary":"done"}', "tool_calls": []}])
        spec = AgentSpec(
            name="structured-agent",
            config_key="structured-agent",
            system_prompt="test",
            tool_names=frozenset(),
            output_schema={
                "type": "object",
                "required": ["summary"],
                "properties": {"summary": {"type": "string"}},
            },
        )
        with patch(
            "agents.loop.get_task_config", return_value={"max_rounds": 1}
        ), patch("agents.loop.get_llm_params", return_value={}), patch(
            "agents.loop.get_llm_service", return_value=llm
        ):
            result = run_agent_loop(spec, [], ToolContext(), [], lambda *_: {})

        self.assertEqual(
            result,
            AgentRunResult.structured_result({"summary": "done"}),
        )

    def test_structured_result_is_unwrapped_for_a_parent_agent(self):
        spec = AgentSpec(
            name="structured-agent",
            config_key="structured-agent",
            system_prompt="test",
            tool_names=frozenset(),
            output_schema={"type": "object"},
        )
        with patch(
            "agents.loop.run_agent_loop",
            return_value=AgentRunResult.structured_result({"summary": "done"}),
        ), patch("lib.llm.config.get_task_config", return_value={}):
            result = spec.run_as_tool('{"query":"question"}', ToolContext())

        self.assertEqual(result, {"summary": "done"})


if __name__ == "__main__":
    unittest.main()
