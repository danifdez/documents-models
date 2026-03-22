"""
Configuration file for models service.

This module contains all configuration parameters for the models service,
including database connections, Qdrant, RAG, and worker settings.

Per-task model configuration is in models.json (see services/model_config.py).
Prompts are in prompts.yaml (see services/prompts.py).
"""

import os
from dotenv import load_dotenv

load_dotenv()

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

# RAG Configuration
RAG_DEFAULT_LIMIT = int(os.getenv("RAG_DEFAULT_LIMIT", "5"))
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "1000"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.35"))

# RAG Chunking Configuration
RAG_CHUNK_TARGET_WORDS = int(os.getenv("RAG_CHUNK_TARGET_WORDS", "150"))
RAG_CHUNK_MAX_WORDS = int(os.getenv("RAG_CHUNK_MAX_WORDS", "250"))
RAG_CHUNK_OVERLAP_WORDS = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "30"))

# File Storage Configuration
DOCUMENTS_STORAGE_DIR = os.path.abspath(os.getenv(
    "DOCUMENTS_STORAGE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "documents")
))

# Worker Configuration
BACKGROUND_HOURS_START = int(os.getenv("BACKGROUND_HOURS_START", "2"))
BACKGROUND_HOURS_END = int(os.getenv("BACKGROUND_HOURS_END", "6"))
