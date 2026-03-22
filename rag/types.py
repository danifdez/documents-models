from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrievedChunk:
    """A single chunk returned from vector search."""
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGContext:
    """Mutable accumulator passed through pipeline stages."""
    # Input
    query: str
    project_id: Optional[str] = None

    # Config
    limit: int = 5
    max_tokens: int = 1000
    score_threshold: Optional[float] = 0.35

    # Retriever output
    chunks: List[RetrievedChunk] = field(default_factory=list)

    # Reranker output
    ranked_chunks: List[RetrievedChunk] = field(default_factory=list)

    # Context builder output
    context_text: str = ""

    # Prompt builder output
    prompt: str = ""

    # Generator output
    response: str = ""
