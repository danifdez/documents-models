import json
import unittest
from unittest.mock import patch

from common.vector_contract import load_vector_candidates, rank_vector_candidates
from tasks.memory.memory import search_memory
from utils.task_dispatch import ensure_task_handler


def embedding(first: float, second: float = 0.0):
    return [first, second, *([0.0] * 382)]


class ArrayResult:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class EmbeddingService:
    def encode_query(self, _query):
        return ArrayResult(embedding(1.0))


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

    @patch("tasks.memory.memory.get_embedding_service")
    def test_memory_search_uses_only_frozen_candidates(self, service):
        service.return_value = EmbeddingService()
        candidates = [
            {
                "id": "7",
                "embedding": embedding(1.0),
                "payload": {
                    "memory_id": 7,
                    "name": "Editor",
                    "type": "fact",
                },
            }
        ]
        result = search_memory(
            {
                "query": "preferred editor",
                "limit": 2,
                "_input_artifacts": {
                    "vector_candidates": json.dumps(
                        {"candidates": candidates}
                    ).encode("utf-8")
                },
            }
        )
        self.assertEqual(result["results"][0]["memoryId"], 7)
        self.assertEqual(result["results"][0]["name"], "Editor")

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


if __name__ == "__main__":
    unittest.main()
