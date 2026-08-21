import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agents.loop import run_agent_loop
from lib.framework.agent import AgentRunResult, AgentSpec
from lib.framework.tool import ToolContext
from lib.execution import get_active_emitter
from lib.execution.emitter import ExecutionEmitter
from tasks.agent_chat.agent_chat import agent_chat
from tasks.assistant_chat.assistant_chat import assistant_chat

from tests.execution.test_emitter import CONTEXT
from tests.execution.support import RecordingExecutionEmitter, RecordingIngestClient


def budget_aware_post(suffix, body):
    return RecordingIngestClient().post(CONTEXT["rootExecutionId"], suffix, body)


class ExactRepeatPolicyParityTest(unittest.TestCase):
    def test_both_chat_handlers_enable_the_same_repeat_policy(self):
        config_path = Path(__file__).resolve().parents[2] / "common" / "tasks.default.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertIs(config["assistant-chat"]["exact_tool_repeat_warning"], True)
        self.assertIs(config["agent-chat"]["exact_tool_repeat_warning"], True)


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
            side_effect=budget_aware_post,
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

    def test_tool_enabled_chat_uses_the_loop_final_text_without_regeneration(self):
        payload = {
            "ownerId": 1,
            "assistantSystem": True,
            "conversation": [{"role": "user", "content": "hello"}],
            "execution": copy.deepcopy(CONTEXT),
        }
        agent = Mock()
        agent.tools.return_value = [{"type": "function"}]
        agent.run.return_value = AgentRunResult.final_text("loop reply")
        generate = Mock(return_value="regenerated reply")

        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": "test-token"}, clear=False), patch(
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
            "tasks.assistant_chat.assistant_chat._memory_for_payload", return_value=[]
        ), patch(
            "tasks.assistant_chat.assistant_chat._extract_memory_action", return_value=None
        ), patch(
            "tasks.assistant_chat.assistant_chat.assistant", agent
        ), patch(
            "tasks.assistant_chat.assistant_chat.generate_reply", generate
        ), patch(
            "lib.execution.emitter.ExecutionEmitter._post",
            side_effect=budget_aware_post,
        ):
            result = assistant_chat(payload)

        self.assertEqual(result["reply"], "loop reply")
        agent.run.assert_called_once()
        generate.assert_not_called()

    def test_tool_budget_partial_preserves_terminal_metadata(self):
        payload = {
            "ownerId": 1,
            "assistantSystem": True,
            "conversation": [{"role": "user", "content": "hello"}],
            "execution": copy.deepcopy(CONTEXT),
        }
        agent = Mock()
        agent.tools.return_value = [{"type": "function"}]
        agent.run.return_value = AgentRunResult.partial_text(
            "partial reply", "tool_budget_exhausted"
        )

        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": "test-token"}, clear=False), patch(
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
            "tasks.assistant_chat.assistant_chat._memory_for_payload", return_value=[]
        ), patch(
            "tasks.assistant_chat.assistant_chat._extract_memory_action", return_value=None
        ), patch(
            "tasks.assistant_chat.assistant_chat.assistant", agent
        ), patch(
            "lib.execution.emitter.ExecutionEmitter._post",
            side_effect=budget_aware_post,
        ):
            result = assistant_chat(payload)

        self.assertEqual(result["reply"], "partial reply")
        self.assertEqual(result["completionKind"], "partial")
        self.assertEqual(result["completionReason"], "tool_budget_exhausted")
        self.assertEqual(result["completionSource"], "model")

    def test_deterministic_partial_preserves_provenance_and_skips_memory(self):
        payload = {
            "ownerId": 1,
            "assistantSystem": True,
            "conversation": [{"role": "user", "content": "hello"}],
            "execution": copy.deepcopy(CONTEXT),
        }
        partial_result = {
            "version": "1",
            "trigger": "closing_output_empty",
            "loopId": CONTEXT["executionId"],
            "grantId": "00000000-0000-4000-8000-000000000011",
            "executionAttemptId": CONTEXT["attemptId"],
            "completedOperations": [{
                "operationId": "00000000-0000-4000-8000-000000000021",
                "toolCallId": "00000000-0000-4000-8000-000000000022",
                "name": "folder_read",
                "summary": "Document read",
            }],
            "pending": ["final_synthesis"],
        }
        agent = Mock()
        agent.tools.return_value = [{"type": "function"}]
        agent.run.return_value = AgentRunResult.deterministic_partial_text(
            "Completed work: Document read",
            "budget_exhausted",
            partial_result,
        )
        extract_memory = Mock(return_value=None)
        ingest = RecordingIngestClient()

        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": "test-token"}, clear=False), patch(
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
            "tasks.assistant_chat.assistant_chat._memory_for_payload", return_value=[]
        ), patch(
            "tasks.assistant_chat.assistant_chat._extract_memory_action", extract_memory
        ), patch(
            "tasks.assistant_chat.assistant_chat.assistant", agent
        ), patch(
            "lib.execution.emitter.ExecutionEmitter._post",
            side_effect=lambda suffix, body: ingest.post(
                CONTEXT["rootExecutionId"], suffix, body
            ),
        ):
            result = assistant_chat(payload)

        self.assertEqual(result["completionKind"], "partial")
        self.assertEqual(result["completionSource"], "runtime_template")
        self.assertEqual(result["partialResult"], partial_result)
        extract_memory.assert_not_called()
        memory_policies = [
            event for event in ingest.sent_events
            if event["eventType"] == "progress.reported"
            and event["payload"].get("kind") == "policy_snapshot"
            and event["payload"].get("policy", {}).get("agentName")
            == "memory_agent"
        ]
        self.assertEqual(memory_policies, [])

    def test_invalid_loop_result_does_not_trigger_hidden_regeneration(self):
        payload = {
            "ownerId": 1,
            "assistantSystem": True,
            "conversation": [{"role": "user", "content": "hello"}],
            "execution": copy.deepcopy(CONTEXT),
        }
        agent = Mock()
        agent.tools.return_value = [{"type": "function"}]
        agent.run.return_value = AgentRunResult.invalid(
            "empty_model_response_after_repair"
        )
        generate = Mock(return_value="hidden retry")

        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": "test-token"}, clear=False), patch(
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
            "tasks.assistant_chat.assistant_chat._memory_for_payload", return_value=[]
        ), patch(
            "tasks.assistant_chat.assistant_chat._extract_memory_action", return_value=None
        ), patch(
            "tasks.assistant_chat.assistant_chat.assistant", agent
        ), patch(
            "tasks.assistant_chat.assistant_chat.generate_reply", generate
        ), patch(
            "lib.execution.emitter.ExecutionEmitter._post",
            side_effect=budget_aware_post,
        ):
            result = assistant_chat(payload)

        self.assertEqual(result["error"], "Model returned an empty response")
        generate.assert_not_called()

    def test_budget_failure_is_returned_with_an_explicit_completion_reason(self):
        payload = {
            "ownerId": 1,
            "assistantSystem": True,
            "conversation": [{"role": "user", "content": "hello"}],
            "execution": copy.deepcopy(CONTEXT),
        }
        agent = Mock()
        agent.tools.return_value = [{"type": "function"}]
        agent.run.return_value = AgentRunResult.invalid(
            "budget_empty_forced_finalization"
        )

        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": "test-token"}, clear=False), patch(
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
            "tasks.assistant_chat.assistant_chat._memory_for_payload", return_value=[]
        ), patch(
            "tasks.assistant_chat.assistant_chat.assistant", agent
        ):
            result = assistant_chat(payload)

        self.assertEqual(result["error"], "budget_empty_forced_finalization")
        self.assertEqual(result["completionReason"], "budget_exhausted")

    def test_final_message_precedes_and_causes_the_memory_phase(self):
        payload = {
            "ownerId": 1,
            "assistantSystem": True,
            "conversation": [{"role": "user", "content": "hello"}],
            "execution": copy.deepcopy(CONTEXT),
        }
        agent = Mock()
        agent.tools.return_value = [{"type": "function"}]

        def run_agent(*_args):
            emitter = get_active_emitter()
            handle = emitter.start_inference("chat_with_tools", {"messages": []})
            emitter.finish_inference(handle, "loop reply", outcome="final_text")
            return AgentRunResult.final_text("loop reply")

        def extract_memory(*_args, **_kwargs):
            emitter = get_active_emitter()
            handle = emitter.start_inference(
                "memory_extraction",
                {"messages": []},
                {"phase": "memory_extraction"},
            )
            emitter.finish_inference(handle, '{"action":"none"}', outcome="final_text")
            return None

        agent.run.side_effect = run_agent
        requests = []

        def post(suffix, body):
            requests.append((suffix, body))
            return {"accepted": len(body.get("events", [])), "duplicates": 0}

        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": "test-token"}, clear=False), patch(
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
            "tasks.assistant_chat.assistant_chat._memory_for_payload", return_value=[]
        ), patch(
            "tasks.assistant_chat.assistant_chat._extract_memory_action",
            side_effect=extract_memory,
        ), patch(
            "tasks.assistant_chat.assistant_chat.assistant", agent
        ), patch(
            "lib.execution.emitter.ExecutionEmitter._post", side_effect=post
        ):
            result = assistant_chat(payload)

        self.assertEqual(result["reply"], "loop reply")
        events = [
            event
            for suffix, body in requests
            if suffix == "events"
            for event in body["events"]
        ]
        response_finish = next(
            event for event in events
            if event["eventType"] == "operation.finished"
            and event["payload"].get("outcome") == "final_text"
        )
        final_message = next(
            event for event in events if event["eventType"] == "message.recorded"
        )
        memory_start = next(
            event for event in events
            if event["eventType"] == "operation.started"
            and event["payload"]["name"] == "memory_extraction"
        )
        self.assertLess(events.index(response_finish), events.index(final_message))
        self.assertLess(events.index(final_message), events.index(memory_start))
        self.assertEqual(final_message["causedByEventId"], response_finish["eventId"])
        self.assertEqual(memory_start["causedByEventId"], final_message["eventId"])


class AgentChatExecutionTest(unittest.TestCase):
    def run_handler(self, tools, loop_result, generated_reply="direct reply"):
        payload = {
            "ownerId": 2,
            "folderScope": "/workspace" if tools else "",
            "conversation": [{"role": "user", "content": "hello"}],
            "execution": copy.deepcopy(CONTEXT),
        }
        agent = Mock()
        agent.tools.return_value = tools
        agent.run.return_value = loop_result
        generate = Mock(return_value=generated_reply)

        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": "test-token"}, clear=False), patch(
            "tasks.agent_chat.agent_chat.get_task_config",
            return_value={"max_tokens": 32, "stream": False},
        ), patch(
            "tasks.agent_chat.agent_chat.get_llm_params", return_value={}
        ), patch(
            "tasks.agent_chat.agent_chat.get_llm_service", return_value=object()
        ), patch(
            "tasks.agent_chat.agent_chat.build_chat_messages",
            return_value=[{"role": "user", "content": "hello"}],
        ), patch(
            "tasks.agent_chat.agent_chat.user_agent_for", return_value=agent
        ), patch(
            "tasks.agent_chat.agent_chat.generate_reply", generate
        ), patch(
            "lib.execution.emitter.ExecutionEmitter._post",
            side_effect=budget_aware_post,
        ):
            result = agent_chat(payload)
        return result, agent, generate

    def test_agent_with_tools_uses_the_loop_final_text_without_regeneration(self):
        result, agent, generate = self.run_handler(
            [{"type": "function"}], AgentRunResult.final_text("loop reply")
        )

        self.assertEqual(result["reply"], "loop reply")
        agent.run.assert_called_once()
        generate.assert_not_called()

    def test_agent_without_an_effective_catalog_uses_one_direct_generation(self):
        result, agent, generate = self.run_handler(
            [], AgentRunResult.final_text("unused loop reply")
        )

        self.assertEqual(result["reply"], "direct reply")
        agent.run.assert_not_called()
        generate.assert_called_once()

    def test_agent_invalid_after_repair_does_not_trigger_hidden_generation(self):
        result, agent, generate = self.run_handler(
            [{"type": "function"}],
            AgentRunResult.invalid("empty_model_response_after_repair"),
            generated_reply="hidden retry",
        )

        self.assertEqual(result["error"], "Model returned an empty response")
        agent.run.assert_called_once()
        generate.assert_not_called()

    def test_agent_budget_failure_has_the_same_terminal_metadata(self):
        result, agent, generate = self.run_handler(
            [{"type": "function"}],
            AgentRunResult.invalid("budget_reservation_consumed"),
        )

        self.assertEqual(result["error"], "budget_reservation_consumed")
        self.assertEqual(result["completionReason"], "budget_exhausted")
        agent.run.assert_called_once()
        generate.assert_not_called()

    def test_agent_tool_budget_partial_preserves_terminal_metadata(self):
        result, agent, generate = self.run_handler(
            [{"type": "function"}],
            AgentRunResult.partial_text(
                "partial reply", "tool_budget_exhausted"
            ),
        )

        self.assertEqual(result["reply"], "partial reply")
        self.assertEqual(result["completionKind"], "partial")
        self.assertEqual(result["completionReason"], "tool_budget_exhausted")
        agent.run.assert_called_once()
        generate.assert_not_called()


class ToolRunTest(unittest.TestCase):
    def emitter(self):
        environment = {
            "EXECUTION_INGEST_TOKEN": "test-token",
        }
        with patch.dict(os.environ, environment, clear=False):
            return RecordingExecutionEmitter(copy.deepcopy(CONTEXT))

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
                emitter.flush_evidence()
                return emitter, result, None
            except Exception as error:  # noqa: BLE001
                return emitter, None, error

    def test_read_tool_has_correlated_start_source_and_finish(self):
        emitter, _result, error = self.run_loop(
            lambda *_: {"summary": "fixture value"})

        self.assertIsNone(error)
        operation = [event for event in emitter.sent_events
                     if event["eventType"].startswith("operation.")]
        source = next(event for event in emitter.sent_events
                      if event["eventType"] == "source.observed")
        self.assertEqual([event["eventType"] for event in operation], [
            "operation.started", "operation.finished",
        ])
        self.assertEqual(operation[0]["operationId"], operation[1]["operationId"])
        self.assertEqual(source["operationId"], operation[0]["operationId"])
        self.assertEqual(operation[1]["causedByEventId"], source["eventId"])
        self.assertEqual(operation[1]["payload"]["status"], "succeeded")
        policy = next(
            event
            for event in emitter.sent_events
            if event["eventType"] == "progress.reported"
        )
        self.assertEqual(policy["payload"]["kind"], "policy_snapshot")
        self.assertEqual(
            operation[0]["payload"]["loopId"],
            policy["payload"]["policy"]["loopId"],
        )
        self.assertEqual(operation[0]["payload"]["agentName"], "test-agent")
        self.assertEqual(operation[0]["payload"]["round"], 1)

    def test_tool_exception_closes_the_operation_as_failed(self):
        def fail(*_args):
            raise RuntimeError("tool unavailable")

        emitter, _result, error = self.run_loop(fail)

        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "tool unavailable")
        finish = next(event for event in emitter.sent_events
                      if event["eventType"] == "operation.finished")
        self.assertEqual(finish["payload"]["status"], "failed")
        self.assertEqual(finish["payload"]["error"]["message"], "tool unavailable")


if __name__ == "__main__":
    unittest.main()
