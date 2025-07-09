from sentence_transformers import SentenceTransformer
from typing import List
import os

class EmbeddingService:
    """Centralized embedding service"""
    
    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.model = SentenceTransformer(self.model_name)
    
    def encode(self, texts: List[str], normalize_embeddings: bool = True):
        """Encode texts to embeddings"""
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)
    
    def encode_single(self, text: str, normalize_embeddings: bool = True):
        """Encode a single text to embedding"""
        return self.model.encode([text], normalize_embeddings=normalize_embeddings)[0]

# Singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service