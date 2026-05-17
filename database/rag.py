import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
    Filter, FieldCondition, MatchValue,
)
from typing import List, Optional
from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_URL,
    QDRANT_COLLECTION, QDRANT_FOLDER_COLLECTION,
)

logger = logging.getLogger(__name__)


class Rag:
    """Thin wrapper around a single Qdrant collection.

    The default collection is the workspace RAG (`rag_docs`) and is filterable by
    `project_id` + `source_id`. A second collection (`assistant_folder_files`)
    holds embeddings for the assistant working folder; it is filterable by
    `assistant_id` + `source_id`. Both share the same Qdrant instance and
    embedding dimension, but are physically separate so a buggy filter cannot
    leak hits across scopes.
    """

    def __init__(self, collection_name: str, payload_indexes: Optional[List[str]] = None):
        self.host = QDRANT_HOST
        self.port = QDRANT_PORT
        self.url = QDRANT_URL
        logger.info("Connecting to Qdrant at %s (collection=%s)", self.url, collection_name)
        self.client = QdrantClient(url=self.url)
        self.collection_name = collection_name
        self.payload_indexes = payload_indexes or ["source_id"]
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                for field in self.payload_indexes:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
        except Exception as e:
            logger.error("Error ensuring collection '%s' exists: %s", self.collection_name, e)
            raise

    def recreate_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            logger.info("Collection '%s' recreated successfully", self.collection_name)
        except Exception as e:
            logger.error("Error recreating collection: %s", e)
            raise

    def upsert_points(self, points: List[PointStruct]) -> bool:
        try:
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
            return True
        except Exception as e:
            logger.error("Error upserting points: %s", e)
            return False

    def query_points(self, query_vector: List[float], limit: int = 3, with_payload: bool = True,
                     project_id: Optional[str] = None, assistant_id: Optional[str] = None,
                     score_threshold: Optional[float] = None):
        try:
            conditions: List[FieldCondition] = []
            if project_id:
                conditions.append(FieldCondition(key="project_id", match=MatchValue(value=str(project_id))))
            if assistant_id:
                conditions.append(FieldCondition(key="assistant_id", match=MatchValue(value=str(assistant_id))))
            query_filter = Filter(must=conditions) if conditions else None
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=with_payload,
                limit=limit,
            ).points
        except Exception as e:
            logger.error("Error querying points: %s", e)
            return []

    def delete_by_source(self, source_id: str) -> bool:
        try:
            delete_filter = Filter(must=[FieldCondition(
                key="source_id", match=MatchValue(value=str(source_id)),
            )])
            self.client.delete(collection_name=self.collection_name, points_selector=delete_filter)
            return True
        except Exception as e:
            logger.error("Error deleting points by source: %s", e)
            return False

    def delete_by_assistant(self, assistant_id: str) -> bool:
        try:
            delete_filter = Filter(must=[FieldCondition(
                key="assistant_id", match=MatchValue(value=str(assistant_id)),
            )])
            self.client.delete(collection_name=self.collection_name, points_selector=delete_filter)
            return True
        except Exception as e:
            logger.error("Error deleting points by assistant: %s", e)
            return False

    def delete_points(self, point_ids: List[str]) -> bool:
        try:
            self.client.delete(collection_name=self.collection_name, points_selector=point_ids)
            return True
        except Exception as e:
            logger.error("Error deleting points: %s", e)
            return False

    def get_collection_info(self):
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error("Error getting collection info: %s", e)
            return None


_rag_database: Optional[Rag] = None
_folder_rag_database: Optional[Rag] = None


def get_rag() -> Rag:
    """Workspace RAG collection (resources / docs / knowledge / notes)."""
    global _rag_database
    if _rag_database is None:
        _rag_database = Rag(
            collection_name=QDRANT_COLLECTION,
            payload_indexes=["project_id", "source_id"],
        )
    return _rag_database


def get_folder_rag() -> Rag:
    """Assistant working-folder collection. Isolated from `rag_docs` so a buggy
    filter cannot leak hits from one scope into the other."""
    global _folder_rag_database
    if _folder_rag_database is None:
        _folder_rag_database = Rag(
            collection_name=QDRANT_FOLDER_COLLECTION,
            payload_indexes=["assistant_id", "source_id"],
        )
    return _folder_rag_database
