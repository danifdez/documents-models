from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_URL, QDRANT_COLLECTION


class Rag:

    def __init__(self):
        self.host = QDRANT_HOST
        self.port = QDRANT_PORT
        self.url = QDRANT_URL
        print(f"Connecting to Qdrant at {self.url}")
        self.client = QdrantClient(url=self.url)
        self.collection_name = QDRANT_COLLECTION

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            if not any(collection.name == self.collection_name for collection in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384, distance=Distance.COSINE),
                )
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise

    def upsert_points(self, points: List[PointStruct]) -> bool:
        """Insert or update points in the collection"""
        try:
            if points:
                self.client.upsert(
                    collection_name=self.collection_name, points=points)
            return True
        except Exception as e:
            print(f"Error upserting points: {e}")
            return False

    def query_points(self, query_vector: List[float], limit: int = 3, with_payload: bool = True):
        """Query similar points from the collection"""
        try:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                with_payload=with_payload,
                limit=limit,
            ).points
        except Exception as e:
            print(f"Error querying points: {e}")
            return []

    def delete_points(self, point_ids: List[str]) -> bool:
        """Delete points by IDs"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            return True
        except Exception as e:
            print(f"Error deleting points: {e}")
            return False

    def get_collection_info(self):
        """Get information about the collection"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None


# Singleton instance
_rag_database = None


def get_rag() -> Rag:
    """Get the singleton Qdrant service instance"""
    global _rag_database
    if _rag_database is None:
        _rag_database = Rag()
    return _rag_database
