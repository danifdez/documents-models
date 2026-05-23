from sentence_transformers import SentenceTransformer
from typing import List
from utils.device import get_device
from services.model_config import get_task_config


class MemoryEmbeddingService:
    """Embedding service for assistant memory entries.

    Separate singleton from the general EmbeddingService (which uses BGE
    English): assistant memory is multilingual (Spanish/English) and goes
    into its own Qdrant collection (``memory_vectors``). E5 asymmetric
    retrieval requires distinct prefixes for queries vs. passages — that
    is what distinguishes this service from the general one.
    """

    def __init__(self):
        task_config = get_task_config("memory-embedding")
        self.model_name = task_config.get("model", "intfloat/multilingual-e5-small")
        self.device = get_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str], normalize_embeddings: bool = True):
        """Encode passages (memory bodies)."""
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


_memory_embedding_service = None


def get_memory_embedding_service() -> MemoryEmbeddingService:
    """Get the singleton memory embedding service instance."""
    global _memory_embedding_service
    if _memory_embedding_service is None:
        _memory_embedding_service = MemoryEmbeddingService()
    return _memory_embedding_service
