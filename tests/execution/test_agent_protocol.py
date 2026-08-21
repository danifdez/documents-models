import unittest
from unittest.mock import patch

from agent.loop import run_one_step
from agent.parse import parse_decision
from agent.types import AgentDefinition, LoopStepOutcome, ModelSpec, StepOutcome
from agents.loop import _normalize_message
from lib.framework.agent import AgentRunResult
from lib.framework.agent_protocol import (
    AgentRunResult as CanonicalAgentRunResult,
    ModelOutcomeKind,
)


class _Llm:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, *_args, **_kwargs):
        return self.reply


class _Database:
    def __init__(self):
        self.result = None
        self.progress = None
        self.statuses = []

    def update_execution_result(self, _execution_id, result, **_kwargs):
        self.result = result
        return True

    def update_execution_status(self, _execution_id, status, **kwargs):
        self.statuses.append((status, kwargs))
        return True

    def update_agent_progress(self, _execution_id, step, checkpoint, **_kwargs):
        self.progress = (step, checkpoint)
        return True


class AgentProtocolConformanceTest(unittest.TestCase):
    def test_both_adapters_produce_the_same_tool_request_semantics(self):
        chat = _normalize_message({
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "function": {
                    "name": "substring_check",
                    "arguments": '{"candidates":["Ada"]}',
                },
            }],
        })
        durable = parse_decision(
            '{"thought":"verify","tool":"substring_check",'
            '"args":{"candidates":["Ada"]}}'
        )

        self.assertEqual(chat.kind, ModelOutcomeKind.TOOL_REQUESTS)
        self.assertEqual(durable.kind, ModelOutcomeKind.TOOL_REQUESTS)
        self.assertEqual(chat.tool_requests[0].name, durable.tool_requests[0].name)
        self.assertEqual(
            chat.tool_requests[0].arguments,
            durable.tool_requests[0].arguments,
        )

    def test_both_adapters_use_canonical_terminal_outcomes(self):
        chat = _normalize_message({"content": "done", "tool_calls": []})
        durable = parse_decision('{"finish":{"answer":"done"}}')

        self.assertEqual(chat.kind, ModelOutcomeKind.FINAL_TEXT)
        self.assertEqual(durable.kind, ModelOutcomeKind.STRUCTURED_RESULT)
        self.assertNotEqual(chat.kind, ModelOutcomeKind.TOOL_REQUESTS)
        self.assertNotEqual(durable.kind, ModelOutcomeKind.TOOL_REQUESTS)

    def test_invalid_structured_arguments_are_typed(self):
        outcome = parse_decision('{"tool":"substring_check","args":[]}')

        self.assertEqual(outcome.kind, ModelOutcomeKind.INVALID)
        self.assertEqual(outcome.reason, "invalid_tool_arguments")

    def test_step_and_run_results_are_the_shared_protocol_types(self):
        self.assertIs(StepOutcome, LoopStepOutcome)
        self.assertIs(AgentRunResult, CanonicalAgentRunResult)

    def test_shared_result_serialization_preserves_partial_metadata(self):
        result = AgentRunResult.deterministic_partial_text(
            "partial",
            "budget_exhausted",
            {"pending": ["synthesis"]},
        ).as_payload()

        self.assertEqual(result["reply"], "partial")
        self.assertEqual(result["completionKind"], "partial")
        self.assertEqual(result["completionReason"], "budget_exhausted")
        self.assertEqual(result["completionSource"], "runtime_template")
        self.assertEqual(result["partialResult"], {"pending": ["synthesis"]})

    def test_durable_runner_serializes_finish_through_shared_result(self):
        execution = {
            "execution_id": "execution-1",
            "task_type": "fixture-agent",
            "payload": {"text": "Ada"},
            "checkpoint": None,
            "step": 0,
            "max_steps": 3,
            "attempt_id": "attempt-1",
        }
        definition = AgentDefinition(
            name="fixture-agent",
            system_prompt="test",
            tools=[],
            model=ModelSpec(path="fixture.gguf"),
        )
        database = _Database()

        with patch(
            "agent.loop.get_llm_for_spec",
            return_value=_Llm('{"finish":{"entities":[]}}'),
        ):
            disposition = run_one_step(execution, definition, database)

        self.assertEqual(disposition, LoopStepOutcome.FINISHED)
        self.assertEqual(database.result["entities"], [])
        self.assertEqual(database.result["_agent"]["reason"], "finish")
        self.assertEqual(database.statuses[-1][1]["phase"], "backend_finalization")

    def test_durable_runner_checkpoints_a_typed_invalid_decision(self):
        execution = {
            "execution_id": "execution-1",
            "task_type": "fixture-agent",
            "payload": {"text": "Ada"},
            "checkpoint": None,
            "step": 0,
            "max_steps": 3,
            "attempt_id": "attempt-1",
        }
        definition = AgentDefinition(
            name="fixture-agent",
            system_prompt="test",
            tools=[],
            model=ModelSpec(path="fixture.gguf"),
        )
        database = _Database()

        with patch(
            "agent.loop.get_llm_for_spec",
            return_value=_Llm('{"tool":"fixture","args":[]}'),
        ):
            disposition = run_one_step(execution, definition, database)

        self.assertEqual(disposition, LoopStepOutcome.CONTINUE)
        self.assertEqual(database.progress[0], 1)
        observation = database.progress[1]["transcript"][-1]["observation"]
        self.assertEqual(observation["error"], "invalid_tool_arguments")


if __name__ == "__main__":
    unittest.main()
