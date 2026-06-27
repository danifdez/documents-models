from sentence_transformers import SentenceTransformer
from typing import List
from utils.device import get_device
from services.model_config import get_task_config


class EmbeddingService:
    """Centralized, multilingual embedding service.

    Single service for every table (workspace RAG, assistant folder files
    and personal memory): they all share the same E5 model and the same 384-dim
    geometry. E5 is asymmetric — passages and queries get distinct
    prefixes (``passage:`` / ``query:``) — so symmetric callers (e.g. dedupe by
    cosine similarity) must compare ``encode`` against ``encode``, never against
    ``encode_query``.
    """

    def __init__(self):
        task_config = get_task_config("embedding")
        self.model_name = task_config.get("model", "intfloat/multilingual-e5-small")
        self.device = get_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str], normalize_embeddings: bool = True):
        """Encode passages (documents) with the E5 passage prefix."""
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=normalize_embeddings)

    def encode_single(self, text: str, normalize_embeddings: bool = True):
        """Encode a single passage."""
        prefixed = f"passage: {text}"
        return self.model.encode([prefixed], normalize_embeddings=normalize_embeddings)[0]

    def encode_query(self, text: str, normalize_embeddings: bool = True):
        """Encode a search query (E5 asymmetric prefix)."""
        prefixed = f"query: {text}"
        return self.model.encode([prefixed], normalize_embeddings=normalize_embeddings)[0]


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
