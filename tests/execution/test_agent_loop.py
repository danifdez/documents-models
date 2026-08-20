import copy
import unittest
from unittest.mock import patch

from agents.loop import run_agent_loop
from lib.execution import InferenceBudgetDenied
from lib.framework.agent import AgentRunResult, AgentSpec
from lib.framework.tool import ToolContext


class FakeLlm:
    def __init__(self, messages, forced_reply=""):
        self.messages = list(messages)
        self.forced_reply = forced_reply
        self.tool_calls = []
        self.chat_calls = []

    def chat_with_tools(self, *_args, **_kwargs):
        self.tool_calls.append((copy.deepcopy(_args), copy.deepcopy(_kwargs)))
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

    def test_repairs_one_empty_output_and_returns_final_text(self):
        llm = FakeLlm([
            {"content": "", "tool_calls": []},
            {"content": "repaired answer", "tool_calls": []},
        ])

        result = self.run_loop(llm)

        self.assertEqual(result, AgentRunResult.final_text("repaired answer"))
        self.assertEqual(len(llm.tool_calls), 2)
        repair_args, repair_kwargs = llm.tool_calls[1]
        self.assertEqual(repair_kwargs["inference_name"], "output_repair")
        self.assertEqual(
            {
                key: repair_kwargs["trace_metadata"][key]
                for key in ("phase", "reason", "attempt", "maxAttempts")
            },
            {
                "phase": "output_repair",
                "reason": "empty_model_response",
                "attempt": 1,
                "maxAttempts": 1,
            },
        )
        self.assertEqual(repair_kwargs["trace_metadata"]["agentName"], "test-agent")
        self.assertEqual(repair_kwargs["trace_metadata"]["round"], 1)
        self.assertEqual(repair_args[0][-1]["role"], "system")
        self.assertIn("previous model output was empty", repair_args[0][-1]["content"])

    def test_repair_can_continue_with_another_tool_call(self):
        llm = FakeLlm([
            {
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "function": {
                        "name": "read_fixture",
                        "arguments": '{"query":"first"}',
                    },
                }],
            },
            {"content": "", "tool_calls": []},
            {
                "content": "",
                "tool_calls": [{
                    "id": "call-2",
                    "function": {
                        "name": "read_fixture",
                        "arguments": '{"query":"second"}',
                    },
                }],
            },
            {"content": "answer from both fixtures", "tool_calls": []},
        ])
        dispatched = []

        result = self.run_loop(
            llm,
            dispatch=lambda name, arguments, _ctx: (
                dispatched.append((name, arguments)) or {"value": "fixture"}
            ),
        )

        self.assertEqual(
            result,
            AgentRunResult.final_text("answer from both fixtures"),
        )
        self.assertEqual(dispatched, [
            ("read_fixture", '{"query":"first"}'),
            ("read_fixture", '{"query":"second"}'),
        ])
        self.assertEqual(
            [kwargs.get("inference_name") for _, kwargs in llm.tool_calls],
            ["chat_with_tools", "chat_with_tools", "output_repair", "chat_with_tools"],
        )
        repair_prompt = llm.tool_calls[2][0][0][-1]["content"]
        final_messages = llm.tool_calls[3][0][0]
        self.assertNotIn(
            repair_prompt,
            [message.get("content") for message in final_messages],
        )

    def test_persistent_empty_output_fails_after_one_repair(self):
        llm = FakeLlm([
            {"content": "", "tool_calls": []},
            {"content": "", "tool_calls": []},
        ])

        result = self.run_loop(llm)

        self.assertEqual(
            result,
            AgentRunResult.invalid("empty_model_response_after_repair"),
        )
        self.assertEqual(len(llm.tool_calls), 2)
        self.assertEqual(llm.tool_calls[1][1]["inference_name"], "output_repair")

    def test_repair_budget_denial_stops_the_loop(self):
        class DeniedRepairLlm(FakeLlm):
            def chat_with_tools(self, *_args, **_kwargs):
                if _kwargs.get("inference_name") == "output_repair":
                    raise InferenceBudgetDenied("budget_hard_limit_reached")
                return super().chat_with_tools(*_args, **_kwargs)

        llm = DeniedRepairLlm([{"content": "", "tool_calls": []}])

        result = self.run_loop(llm)

        self.assertEqual(
            result,
            AgentRunResult.invalid("budget_hard_limit_reached"),
        )

    def test_later_empty_output_does_not_receive_a_second_repair(self):
        llm = FakeLlm([
            {"content": "", "tool_calls": []},
            {
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "function": {"name": "read_fixture", "arguments": "{}"},
                }],
            },
            {"content": "", "tool_calls": []},
        ])

        result = self.run_loop(llm)

        self.assertEqual(
            result,
            AgentRunResult.invalid("empty_model_response_after_repair"),
        )
        self.assertEqual(len(llm.tool_calls), 3)
        self.assertEqual(
            sum(
                kwargs.get("inference_name") == "output_repair"
                for _, kwargs in llm.tool_calls
            ),
            1,
        )
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

        self.assertEqual(
            result,
            AgentRunResult.partial_text("budget answer", "budget_exhausted"),
        )
        self.assertEqual(len(llm.chat_calls), 1)
        self.assertEqual(llm.chat_calls[0][1]["inference_name"], "forced_finalization")
        self.assertEqual(
            {
                key: llm.chat_calls[0][1]["trace_metadata"][key]
                for key in ("phase", "reason", "round", "maxRounds")
            },
            {
                "phase": "forced_finalization",
                "reason": "step_budget_exhausted",
                "round": 1,
                "maxRounds": 1,
            },
        )

    def test_empty_forced_finalization_is_not_repaired(self):
        tool_request = {
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "read_fixture", "arguments": "{}"},
            }],
        }
        llm = FakeLlm([tool_request], forced_reply="")

        result = self.run_loop(llm, max_rounds=1)

        self.assertEqual(
            result,
            AgentRunResult.invalid("budget_empty_forced_finalization"),
        )
        self.assertEqual(len(llm.tool_calls), 1)
        self.assertEqual(len(llm.chat_calls), 1)

    def test_consumed_closing_reservation_fails_without_another_inference(self):
        class ConsumedClosingLlm(FakeLlm):
            def chat(self, *_args, **_kwargs):
                raise InferenceBudgetDenied("budget_reservation_consumed")

        tool_request = {
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "read_fixture", "arguments": "{}"},
            }],
        }
        llm = ConsumedClosingLlm([tool_request])

        result = self.run_loop(llm, max_rounds=1)

        self.assertEqual(
            result,
            AgentRunResult.invalid("budget_reservation_consumed"),
        )
        self.assertEqual(len(llm.tool_calls), 1)

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

    def test_structured_agent_empty_output_uses_its_existing_schema_path(self):
        llm = FakeLlm(
            [{"content": "", "tool_calls": []}],
            forced_reply='{"summary":"repaired schema"}',
        )
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
            AgentRunResult.structured_result({"summary": "repaired schema"}),
        )
        self.assertEqual(len(llm.tool_calls), 1)
        self.assertFalse(any(
            kwargs.get("inference_name") == "output_repair"
            for _, kwargs in llm.tool_calls
        ))

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
