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
        "tasks.relationship_extraction.relationship_extraction._get_relationship_pipeline"
    )
    def test_map_classifies_only_configured_relationships(self, get_pipeline):
        classifier = Mock()

        def classify(_context, **kwargs):
            return [
                {"labels": ["created"], "scores": [0.94]},
                {"labels": ["created"], "scores": [0.1]},
            ]

        classifier.side_effect = classify
        get_pipeline.return_value = classifier

        result = relationship_extraction_map(
            {
                "content": "Ada Lovelace created the Analytical Engine.",
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
            result["relationships"][0]["predicate"], "created"
        )
        self.assertGreater(result["relationships"][0]["confidence"], 0.99)
        classifier.assert_called_once()

    @patch(
        "tasks.relationship_extraction.relationship_extraction._get_relationship_pipeline"
    )
    def test_map_rejects_scores_below_the_configured_threshold(self, get_pipeline):
        classifier = Mock(
            return_value=[
                {"labels": ["works for"], "scores": [0.79]},
                {"labels": ["works for"], "scores": [0.1]},
            ]
        )
        get_pipeline.return_value = classifier

        result = relationship_extraction_map(
            {
                "content": "Ada Lovelace worked for Babbage Ltd.",
                "entities": [
                    {"name": "Ada Lovelace", "type": "PERSON"},
                    {"name": "Babbage Ltd", "type": "ORG"},
                ],
            }
        )

        self.assertEqual(result, {"relationships": []})

    @patch(
        "tasks.relationship_extraction.relationship_extraction._get_relationship_pipeline"
    )
    def test_map_rejects_high_prior_scores_without_contextual_evidence(
        self, get_pipeline
    ):
        classifier = Mock(
            side_effect=[
                [
                    {"labels": ["owns"], "scores": [0.9]},
                    {"labels": ["owns"], "scores": [0.5]},
                ],
                [
                    {"labels": ["is part of"], "scores": [0.9]},
                    {"labels": ["is part of"], "scores": [0.5]},
                ],
            ]
        )
        get_pipeline.return_value = classifier

        result = relationship_extraction_map(
            {
                "content": "Acme mentioned Globex in its report.",
                "entities": [
                    {"name": "Acme", "type": "ORG"},
                    {"name": "Globex", "type": "ORG"},
                ],
            }
        )

        self.assertEqual(result, {"relationships": []})

    @patch(
        "tasks.relationship_extraction.relationship_extraction._get_relationship_pipeline"
    )
    def test_map_collapses_reverse_symmetric_relationships(self, get_pipeline):
        classifier = Mock(
            side_effect=[
                [
                    {"labels": ["is partnered with"], "scores": [0.95]},
                    {"labels": ["is partnered with"], "scores": [0.5]},
                ],
                [
                    {"labels": ["is partnered with"], "scores": [0.96]},
                    {"labels": ["is partnered with"], "scores": [0.5]},
                ],
            ]
        )
        get_pipeline.return_value = classifier

        result = relationship_extraction_map(
            {
                "content": "Acme is partnered with Globex.",
                "entities": [
                    {"name": "Acme", "type": "ORG"},
                    {"name": "Globex", "type": "ORG"},
                ],
            }
        )

        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(
            {
                key: value
                for key, value in result["relationships"][0].items()
                if key != "confidence"
            },
            {
                "subject": "Acme",
                "predicate": "partnered_with",
                "object": "Globex",
                "context": "Acme is partnered with Globex.",
            },
        )
        self.assertAlmostEqual(result["relationships"][0]["confidence"], 0.96)

    @patch(
        "tasks.relationship_extraction.relationship_extraction._get_relationship_pipeline"
    )
    def test_map_skips_entities_absent_from_the_chunk(self, get_pipeline):
        classifier = Mock()
        get_pipeline.return_value = classifier

        result = relationship_extraction_map(
            {
                "content": "Ada Lovelace wrote notes.",
                "entities": [
                    {"name": "Ada Lovelace", "type": "PERSON"},
                    {"name": "Analytical Engine", "type": "PRODUCT"},
                ],
            }
        )

        self.assertEqual(result, {"relationships": []})
        classifier.assert_not_called()

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
