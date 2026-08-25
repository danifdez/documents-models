import json
import unittest
from unittest.mock import Mock, patch

from common.execution_registry import TASK_HANDLERS
from lib.execution.step_executor import execute_assignment
from tasks.relationship_extraction.relationship_extraction import (
    relationship_extraction_map,
    relationship_extraction_reduce,
)


def _relationship(subject, predicate, obj, confidence=0.5):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "confidence": confidence,
        "context": "",
    }


class RelationshipExtractionStepTest(unittest.TestCase):
    def test_registers_only_durable_relationship_extraction_capabilities(self):
        self.assertIn("relationship-extraction-map", TASK_HANDLERS)
        self.assertIn("relationship-extraction-reduce", TASK_HANDLERS)
        self.assertNotIn("relationship-extraction", TASK_HANDLERS)
        self.assertNotIn("relationship-query", TASK_HANDLERS)
        self.assertNotIn("relationship-modify", TASK_HANDLERS)

    @patch(
        "tasks.relationship_extraction.relationship_extraction.get_llm_params",
        return_value={},
    )
    @patch(
        "tasks.relationship_extraction.relationship_extraction.get_llm_service"
    )
    def test_map_runs_one_inference_for_one_bounded_chunk(
        self, get_llm_service, _get_llm_params
    ):
        llm = Mock()
        llm.chat.return_value = json.dumps(
            [
                {
                    "subject": "Ada Lovelace",
                    "predicate": "Documented",
                    "object": "Analytical Engine",
                }
            ]
        )
        get_llm_service.return_value = llm

        result = relationship_extraction_map(
            {
                "content": "Ada Lovelace documented the Analytical Engine.",
                "entities": [
                    {"id": 1, "name": "Ada Lovelace", "type": "PERSON"},
                    {
                        "id": 2,
                        "name": "Analytical Engine",
                        "type": "PRODUCT",
                    },
                ],
            }
        )

        self.assertEqual(
            result["relationships"][0]["predicate"], "documented"
        )
        self.assertEqual(result["relationships"][0]["confidence"], 0.5)
        llm.chat.assert_called_once()

    def test_reduce_deduplicates_in_map_order_and_keeps_best_confidence(self):
        low = _relationship("Ada", "created", "Notes", 0.4)
        high = _relationship("Ada", "created", "Notes", 0.9)
        other = _relationship("Notes", "describes", "Engine", 0.5)

        result = relationship_extraction_reduce(
            {"partials": [[low], [], [other, high]]}
        )

        self.assertEqual(result["relationships"], [high, other])

    def test_map_rejects_an_unbounded_entity_list(self):
        entities = [
            {"id": index, "name": f"Entity {index}", "type": "ORG"}
            for index in range(201)
        ]

        with self.assertRaisesRegex(ValueError, "too many entities"):
            relationship_extraction_map(
                {"content": "Entity 1 works with Entity 2.", "entities": entities}
            )

    def test_map_failure_becomes_a_failed_inference_result(self):
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {
                "taskType": "relationship-extraction-map",
                "payload": {"content": "", "entities": []},
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["output"]["outcome"], {"kind": "failed"})

    def test_reduce_executes_as_replayable_code_assignment(self):
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "code",
            "work": {
                "taskType": "relationship-extraction-reduce",
                "payload": {"partials": [[]]},
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            result["output"],
            {"kind": "code", "value": {"relationships": []}},
        )


if __name__ == "__main__":
    unittest.main()
