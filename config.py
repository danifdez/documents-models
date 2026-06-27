"""
Configuration file for models service.

All configuration is loaded from config/config.json (see services/model_config.py).
This module re-exports infrastructure constants for backward compatibility.
"""

from services.model_config import get_config

_cfg = get_config()

# PostgreSQL Database Configuration
_db = _cfg.get("database", {})
POSTGRES_HOST = _db.get("host", "localhost")
POSTGRES_PORT = int(_db.get("port", 5432))
POSTGRES_DB = _db.get("name", "documents")
POSTGRES_USER = _db.get("user", "postgres")
POSTGRES_PASSWORD = _db.get("password", "example")
JOBS_TABLE = _db.get("jobs_table", "jobs")

# Vector store (pgvector) — embeddings live in PostgreSQL tables created by the
# backend migrations. One table per scope (physically isolated, as before).
_vec = _cfg.get("vectors", {})
RAG_TABLE = _vec.get("rag_table", "rag_chunks")
FOLDER_TABLE = _vec.get("folder_table", "indexed_file_chunks")
MEMORY_TABLE = _vec.get("memory_table", "memory_vectors")

# Neo4j Configuration
_n4j = _cfg.get("neo4j", {})
NEO4J_ENABLED = _n4j.get("enabled", False)
NEO4J_HOST = _n4j.get("host", "localhost")
NEO4J_PORT = int(_n4j.get("port", 7687))
NEO4J_URI = f"bolt://{NEO4J_HOST}:{NEO4J_PORT}"
NEO4J_USER = _n4j.get("user", "neo4j")
NEO4J_PASSWORD = _n4j.get("password", "example123")

