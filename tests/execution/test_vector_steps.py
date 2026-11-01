import json
import base64
import unittest
from unittest.mock import patch

from common.vector_contract import (
    load_vector_candidates,
    rank_vector_candidates,
    vector_points_for_chunks,
    vector_points_output,
)
from common.execution_registry import TASK_HANDLERS
from lib.execution.step_executor import execute_assignment
from utils.task_dispatch import ensure_task_handler


def embedding(first: float, second: float = 0.0):
    return [first, second, *([0.0] * 382)]


class _Embedding(list):
    """Stand-in for the numpy rows returned by the embedding service."""

    def tolist(self):
        return list(self)


class VectorContractTest(unittest.TestCase):
    def test_loads_candidates_from_assignment_artifact(self):
        candidates = [
            {"id": "one", "embedding": embedding(1.0), "payload": {"text": "A"}}
        ]
        loaded = load_vector_candidates(
            {
                "_input_artifacts": {
                    "vector_candidates": json.dumps(
                        {"candidates": candidates}
                    ).encode("utf-8")
                }
            }
        )
        self.assertEqual(loaded, candidates)

    def test_ranks_candidates_without_database_access(self):
        candidates = [
            {"id": "low", "embedding": embedding(0.0, 1.0), "payload": {}},
            {"id": "high", "embedding": embedding(1.0), "payload": {}},
        ]
        ranked = rank_vector_candidates(embedding(1.0), candidates, 2)
        self.assertEqual([item["id"] for item in ranked], ["high", "low"])
        self.assertAlmostEqual(ranked[0]["score"], 1.0)

    def test_builds_one_point_per_chunk_with_source_metadata(self):
        points = vector_points_for_chunks(
            "doc_7",
            ["First", "Second"],
            [_Embedding(embedding(1.0)), _Embedding(embedding(0.0, 1.0))],
            {"doc_id": 7},
        )

        self.assertEqual([point["id"] for point in points], ["doc_7:1", "doc_7:2"])
        self.assertEqual(
            points[0]["payload"],
            {
                "doc_id": 7,
                "text": "First",
                "source_id": "doc_7",
                "part_number": 1,
                "total_chunks": 2,
            },
        )
        self.assertEqual(points[1]["embedding"], embedding(0.0, 1.0))

    def test_chunk_identity_cannot_be_overridden_by_metadata(self):
        points = vector_points_for_chunks(
            "doc_7",
            ["First"],
            [_Embedding(embedding(1.0))],
            {"text": "Spoofed", "source_id": "other"},
        )

        self.assertEqual(points[0]["payload"]["text"], "First")
        self.assertEqual(points[0]["payload"]["source_id"], "doc_7")

    def test_rejects_mismatched_chunk_and_embedding_counts(self):
        with self.assertRaisesRegex(ValueError, "counts must match"):
            vector_points_for_chunks(
                "doc_7",
                ["First", "Second"],
                [_Embedding(embedding(1.0))],
            )

    def test_rejects_missing_candidates_artifact(self):
        with self.assertRaisesRegex(ValueError, "artifact is required"):
            load_vector_candidates({})

    def test_removed_storage_tasks_cannot_be_dispatched(self):
        for task_type in (
            "delete-vectors",
            "memory-delete-vectors",
            "indexed-file-delete-vectors",
        ):
            with self.subTest(task_type=task_type):
                self.assertFalse(ensure_task_handler(task_type))

    def test_vector_points_leave_the_structured_result_as_artifacts(self):
        task_type = "vector-artifact-test"
        points = [
            {
                "id": f"resource_1:{index}",
                "embedding": embedding(1.0),
                "payload": {"text": f"Chunk {index}"},
            }
            for index in range(1, 258)
        ]
        TASK_HANDLERS[task_type] = lambda _payload: vector_points_output(
            {"sourceId": "resource_1", "chunks": 257}, points
        )
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "service",
            "work": {"taskType": task_type, "payload": {}},
        }
        uploads = []
        try:
            with patch(
                "lib.execution.step_executor.ensure_task_handler",
                return_value=True,
            ):
                result = execute_assignment(
                    assignment, output_artifacts=uploads
                )
        finally:
            TASK_HANDLERS.pop(task_type, None)

        self.assertEqual(
            result["output"]["value"],
            {"sourceId": "resource_1", "chunks": 257, "pointCount": 257},
        )
        self.assertNotIn("points", result["output"]["value"])
        self.assertEqual(len(result["artifactRefs"]), 2)
        self.assertEqual(
            [ref["revision"] for ref in result["artifactRefs"]], [1, 2]
        )
        documents = [
            json.loads(base64.b64decode(upload["bodyBase64"]))
            for upload in uploads
        ]
        self.assertEqual(
            [len(document["points"]) for document in documents], [256, 1]
        )
        self.assertEqual(
            documents[0]["points"] + documents[1]["points"], points
        )


if __name__ == "__main__":
    unittest.main()
