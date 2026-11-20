import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from common.vector_contract import EMBEDDING_DIMENSIONS
from rag import generator, retriever
from services import embedding_service
from tasks.ask.ask import ask_question
from tasks.indexed_file import indexed_file
from tasks.ingest import ingest
from tools import chunk_text

QUERY_VECTOR = [1] + [0] * (EMBEDDING_DIMENSIONS - 1)
OTHER_VECTOR = [0, 1] + [0] * (EMBEDDING_DIMENSIONS - 2)


def candidates_artifact(*candidates):
    return {"vector_candidates": json.dumps({"candidates": candidates}).encode()}


def candidate(point_id, text, embedding=QUERY_VECTOR, **metadata):
    return {"id": point_id, "embedding": embedding, "payload": {"text": text, **metadata}}


class IndexingAssignmentTests(unittest.TestCase):
    def test_ingest_records_the_source_identity_and_normalized_embedding(self):
        cases = (
            ({}, {"resourceId": 7}, "resource_7", "resource"),
            ({"sourceType": "doc"}, {"docId": 9}, "doc_9", "doc"),
            ({"sourceType": "knowledge"}, {"knowledgeEntryId": 2}, "knowledge_2", "knowledge"),
        )
        for source, identity, source_id, source_type in cases:
            with self.subTest(source=source):
                with patch.object(ingest, "get_embedding_service") as get_service:
                    encoder = get_service.return_value
                    encoder.encode.return_value = np.array([QUERY_VECTOR])
                    result = ingest.ingest({
                        **source, **identity, "content": "<p>Rainfall in May</p>", "projectId": 3,
                    })

                encoder.encode.assert_called_once_with(["Rainfall in May"], normalize_embeddings=True)
                self.assertEqual(result.value, {"sourceId": source_id, "chunks": 1, "pointCount": 1})
                point = json.loads(result.artifacts[0].body)["points"][0]
                self.assertEqual(point["id"], f"{source_id}:1")
                self.assertEqual(point["embedding"], QUERY_VECTOR)
                self.assertEqual(point["payload"], {
                    "source_id": source_id, "source_type": source_type, "project_id": 3,
                    "part_number": 1, "total_chunks": 1, "text": "Rainfall in May",
                })

    def test_ingest_rejects_invalid_payload_and_skips_empty_documents(self):
        with patch.object(ingest, "get_embedding_service") as get_service:
            with self.assertRaisesRegex(ValueError, "Unsupported source type"):
                ingest.ingest({"sourceType": "other", "content": "x"})
            with self.assertRaisesRegex(ValueError, "content must be a string"):
                ingest.ingest({"resourceId": 7, "content": []})
            result = ingest.ingest({"resourceId": 7, "content": "<p> </p><script>skip</script>"})

        get_service.assert_not_called()
        self.assertEqual(result.value, {"sourceId": "resource_7", "chunks": 0, "pointCount": 0})
        self.assertEqual(result.artifacts, [])

    def test_indexed_file_ingest_returns_a_vector_artifact_with_identity_and_checksum(self):
        with patch.object(indexed_file, "get_embedding_service") as get_service:
            encoder = get_service.return_value
            encoder.encode.return_value = np.array([QUERY_VECTOR])
            empty = indexed_file.ingest_indexed_file({"indexedFileId": 5, "content": " "})
            with self.assertRaisesRegex(ValueError, "content must be a string"):
                indexed_file.ingest_indexed_file({"indexedFileId": 5})
            result = indexed_file.ingest_indexed_file({
                "indexedFileId": "5", "content": " File text ", "filename": "notes.txt", "checksum": "digest",
            })

        self.assertEqual(empty.value["pointCount"], 0)
        self.assertEqual(empty.artifacts, [])
        encoder.encode.assert_called_once_with(["File text"], normalize_embeddings=True)
        self.assertEqual(result.value, {
            "indexedFileId": 5, "sourceId": "indexed_file_5", "chunks": 1, "pointCount": 1, "checksum": "digest",
        })
        point = json.loads(result.artifacts[0].body)["points"][0]
        self.assertEqual(point["payload"]["indexed_file_id"], 5)
        self.assertEqual(point["payload"]["filename"], "notes.txt")

    def test_search_groups_chunks_by_file_and_uses_the_best_matching_snippet(self):
        long_text = "Sample " * 50
        artifact = candidates_artifact(
            candidate("a", long_text, indexed_file_id=5, filename="notes.txt"),
            candidate("b", "Repeated source", indexed_file_id=5, filename="notes.txt"),
            candidate("c", "No file identity"),
            candidate("d", "Second file", indexed_file_id=6, filename="second.txt"),
            candidate("e", "Below score threshold", OTHER_VECTOR, indexed_file_id=7, filename="third.txt"),
        )
        with patch.object(indexed_file, "get_embedding_service") as get_service:
            encoder = get_service.return_value
            encoder.encode_query.return_value = np.array(QUERY_VECTOR)
            self.assertEqual(indexed_file.search_indexed_files({"query": " "}), {"results": []})
            result = indexed_file.search_indexed_files({
                "query": " search ", "limit": 2, "score_threshold": "0.5", "_input_artifacts": artifact,
            })

        encoder.encode_query.assert_called_once_with("search")
        self.assertEqual(result, {"results": [
            {"indexedFileId": 5, "filename": "notes.txt", "snippet": long_text[:300] + "…", "score": 1.0},
            {"indexedFileId": 6, "filename": "second.txt", "snippet": "Second file", "score": 1.0},
        ]})


class AskAssignmentTests(unittest.TestCase):
    def test_ask_uses_local_candidates_and_records_the_ranking_in_the_output_artifact(self):
        artifact = candidates_artifact(
            candidate("a", "Rain fell in May.", source_id="doc_1"),
            candidate("b", "Rain fell in May.", source_id="duplicate"),
            candidate("c", "", source_id="empty"),
            candidate("d", "Off topic", OTHER_VECTOR, source_id="irrelevant"),
        )
        with (
            patch.object(retriever, "get_embedding_service") as get_embedding,
            patch.object(generator, "get_llm_service") as get_llm,
        ):
            get_embedding.return_value.encode_query.return_value = np.array(QUERY_VECTOR)
            llm = get_llm.return_value
            llm.ask.return_value = "It rained in May."
            result = ask_question({
                "question": "When did it rain?", "projectId": 3, "context": " Weather station A ",
                "graphContext": [
                    {"source": "May", "predicate": "recorded", "target": "rain"},
                    {"source": "incomplete"},
                ],
                "_input_artifacts": artifact,
            })

        get_embedding.return_value.encode_query.assert_called_once_with("When did it rain?")
        prompt = llm.ask.call_args.args[0]
        self.assertEqual(prompt.count("Rain fell in May."), 1)
        self.assertIn("Weather station A", prompt)
        self.assertIn("May --[recorded]--> rain", prompt)
        self.assertIn("When did it rain?", prompt)
        self.assertNotIn("Off topic", prompt)
        self.assertNotIn("incomplete", prompt)
        self.assertEqual(result.value, {"response": "It rained in May."})
        output = result.artifacts[0]
        self.assertEqual((output.role, output.kind), ("rag_context", "rag_context_evidence"))
        evidence = json.loads(output.body)
        self.assertEqual(evidence["candidateCount"], 4)
        self.assertEqual(evidence["relationshipCount"], 2)
        self.assertEqual([chunk["sourceId"] for chunk in evidence["retrieved"]], ["doc_1", "duplicate", "empty"])
        self.assertEqual(evidence["ranked"], [{
            "sourceId": "doc_1",
            "textSha256": "sha256:" + hashlib.sha256(b"Rain fell in May.").hexdigest(),
            "score": "1.0",
        }])

    def test_empty_generation_returns_the_standard_no_evidence_response(self):
        with (
            patch.object(retriever, "get_embedding_service") as get_embedding,
            patch.object(generator, "get_llm_service") as get_llm,
        ):
            get_embedding.return_value.encode_query.return_value = np.array(QUERY_VECTOR)
            get_llm.return_value.ask.return_value = ""
            result = ask_question({"question": "Unknown?", "_input_artifacts": candidates_artifact()})

        self.assertEqual(result.value, {"response": "No relevant information was found to answer this question."})
        self.assertEqual(json.loads(result.artifacts[0].body)["ranked"], [])


class EmbeddingServiceTests(unittest.TestCase):
    def test_query_and_passage_embedders_use_different_prefixes_and_one_model(self):
        with (
            patch.object(embedding_service, "_embedding_service", None),
            patch.object(embedding_service, "get_task_config", return_value={"model": "example/e5"}),
            patch.object(embedding_service, "get_device", return_value="cpu"),
            patch.object(embedding_service, "SentenceTransformer") as loader,
        ):
            model = loader.return_value
            model.encode.return_value = np.array([QUERY_VECTOR])
            service = embedding_service.get_embedding_service()
            self.assertIs(service, embedding_service.get_embedding_service())
            loader.assert_called_once_with("example/e5", device="cpu")

            for method, argument, inputs, expected in (
                (service.encode, ["a", "b"], ["passage: a", "passage: b"], [QUERY_VECTOR]),
                (service.encode_single, "a", ["passage: a"], QUERY_VECTOR),
                (service.encode_query, "a", ["query: a"], QUERY_VECTOR),
            ):
                with self.subTest(method=method.__name__):
                    self.assertEqual(method(argument, normalize_embeddings=False).tolist(), expected)
                    model.encode.assert_called_with(inputs, normalize_embeddings=False)


class ChunkToolTests(unittest.TestCase):
    request: ClassVar[dict] = {"format": "html", "text": "<p>Hello</p>", "target_words": 3, "max_words": 5, "overlap_words": 1}

    def run_tool(self, body):
        output = io.StringIO()
        with patch.object(chunk_text.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(body))), redirect_stdout(output):
            chunk_text.main()
        return json.loads(output.getvalue())

    def test_reads_html_and_plain_text_requests(self):
        self.assertEqual(self.run_tool(json.dumps(self.request).encode()), {"blocks": ["Hello"], "chunks": ["Hello"]})
        text = {**self.request, "format": "text"}
        self.assertEqual(self.run_tool(json.dumps(text).encode()), {"blocks": ["<p>Hello</p>"], "chunks": ["<p>Hello</p>"]})

    def test_rejects_wrong_size_fields_format_and_limits(self):
        cases = [
            (b"", "input size"),
            (b"x" * (chunk_text.MAX_REQUEST_BYTES + 1), "input size"),
            (b"{}", "fields"),
            (json.dumps({**self.request, "format": "xml"}).encode(), "options"),
            (json.dumps({**self.request, "target_words": True}).encode(), "options"),
            (json.dumps({**self.request, "overlap_words": 2}).encode(), "options"),
        ]
        for body, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.run_tool(body)
