from sentence_transformers import SentenceTransformer
from typing import List
from utils.device import get_device
from services.model_config import get_task_config


class EmbeddingService:
    """Centralized embedding service"""

    def __init__(self):
        task_config = get_task_config("embedding")
        self.model_name = task_config.get("model", "BAAI/bge-small-en-v1.5")
        self.device = get_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str], normalize_embeddings: bool = True):
        """Encode texts to embeddings"""
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)

    def encode_single(self, text: str, normalize_embeddings: bool = True):
        """Encode a single text to embedding"""
        return self.model.encode([text], normalize_embeddings=normalize_embeddings)[0]

    def encode_query(self, text: str, normalize_embeddings: bool = True):
        """Encode a query with BGE instruction prefix for asymmetric retrieval."""
        prefixed = "Represent this sentence: " + text
        return self.model.encode([prefixed], normalize_embeddings=normalize_embeddings)[0]


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
