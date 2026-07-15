"""
Configuration file for models service.

All configuration is loaded from config/config.json (see services/model_config.py).
This module re-exports infrastructure constants for backward compatibility.
"""

from lib.llm.config import get_config

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

# Graph Configuration (Apache AGE — openCypher over the same PostgreSQL instance)
_graph = _cfg.get("graph", {})
GRAPH_NAME = _graph.get("name", "documents")
# The graph is the storage layer for the "relationships" feature, so it follows
# that feature flag. `graph.enabled` can still force it off independently.
_relationships_on = _cfg.get("features", {}).get("relationships", True)
GRAPH_ENABLED = _relationships_on and _graph.get("enabled", True)
# Multi-hop neighborhood traversal used for GraphRAG.
GRAPH_NEIGHBORHOOD_DEPTH = int(_graph.get("neighborhood_depth", 2))
GRAPH_NEIGHBORHOOD_LIMIT = int(_graph.get("neighborhood_limit", 50))

