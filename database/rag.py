import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType, Filter, FieldCondition, MatchValue
from typing import List, Optional
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_URL, QDRANT_COLLECTION

logger = logging.getLogger(__name__)


class Rag:

    def __init__(self):
        self.host = QDRANT_HOST
        self.port = QDRANT_PORT
        self.url = QDRANT_URL
        logger.info("Connecting to Qdrant at %s", self.url)
        self.client = QdrantClient(url=self.url)
        self.collection_name = QDRANT_COLLECTION

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist, with payload index on project_id"""
        try:
            collections = self.client.get_collections().collections
            if not any(collection.name == self.collection_name for collection in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384, distance=Distance.COSINE),
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="project_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        except Exception as e:
            logger.error("Error ensuring collection exists: %s", e)
            raise

    def recreate_collection(self):
        """Drop and recreate collection. Use for migration after schema changes."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            logger.info("Collection '%s' recreated successfully", self.collection_name)
        except Exception as e:
            logger.error("Error recreating collection: %s", e)
            raise

    def upsert_points(self, points: List[PointStruct]) -> bool:
        """Insert or update points in the collection"""
        try:
            if points:
                self.client.upsert(
                    collection_name=self.collection_name, points=points)
            return True
        except Exception as e:
            logger.error("Error upserting points: %s", e)
            return False

    def query_points(self, query_vector: List[float], limit: int = 3, with_payload: bool = True,
                     project_id: Optional[str] = None, score_threshold: Optional[float] = None):
        """Query similar points from the collection, optionally filtered by project."""
        try:
            query_filter = None
            if project_id:
                query_filter = Filter(
                    must=[FieldCondition(
                        key="project_id",
                        match=MatchValue(value=str(project_id))
                    )]
                )
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
        """Delete all points for a given source_id (resource or doc)."""
        try:
            delete_filter = Filter(
                must=[FieldCondition(
                    key="source_id",
                    match=MatchValue(value=str(source_id))
                )]
            )
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter,
            )
            return True
        except Exception as e:
            logger.error("Error deleting points by source: %s", e)
            return False

    def delete_points(self, point_ids: List[str]) -> bool:
        """Delete points by IDs"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            return True
        except Exception as e:
            logger.error("Error deleting points: %s", e)
            return False

    def get_collection_info(self):
        """Get information about the collection"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error("Error getting collection info: %s", e)
            return None


# Singleton instance
_rag_database = None


def get_rag() -> Rag:
    """Get the singleton Qdrant service instance"""
    global _rag_database
    if _rag_database is None:
        _rag_database = Rag()
    return _rag_database
