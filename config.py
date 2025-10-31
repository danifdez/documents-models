"""
Configuration file for models service.

This module contains all configuration parameters for the models service,
including paths to AI models, database connections, and service settings.
"""

import os

# PostgreSQL Database Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "database")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "documents")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "example")
JOBS_TABLE = os.getenv("JOBS_TABLE", "jobs")

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_docs")

# LLM Model Configuration
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "Mistral-7B-Instruct-v0.3-Q8_0.gguf"
)
LLM_MODEL_PATH = f"/app/models/{LLM_MODEL_NAME}"

# LLM Parameters
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "32768"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "4"))
LLM_N_BATCH = int(os.getenv("LLM_N_BATCH", "64"))

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-small-en-v1.5"
)

# RAG Configuration
RAG_DEFAULT_LIMIT = int(os.getenv("RAG_DEFAULT_LIMIT", "5"))
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "1000"))

# File Storage Configuration
DOCUMENTS_STORAGE_DIR = os.getenv(
    "DOCUMENTS_STORAGE_DIR", "/app/documents_storage")
