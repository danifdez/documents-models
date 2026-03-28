"""
Configuration file for models service.

All configuration is loaded from config/config.json (see services/model_config.py).
This module re-exports infrastructure constants for backward compatibility.
"""

import os
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

# Qdrant Configuration
_qd = _cfg.get("qdrant", {})
QDRANT_ENABLED = _qd.get("enabled", True)
QDRANT_HOST = _qd.get("host", "localhost")
QDRANT_PORT = int(_qd.get("port", 6333))
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
QDRANT_COLLECTION = _qd.get("collection", "rag_docs")

# Neo4j Configuration
_n4j = _cfg.get("neo4j", {})
NEO4J_ENABLED = _n4j.get("enabled", False)
NEO4J_HOST = _n4j.get("host", "localhost")
NEO4J_PORT = int(_n4j.get("port", 7687))
NEO4J_URI = f"bolt://{NEO4J_HOST}:{NEO4J_PORT}"
NEO4J_USER = _n4j.get("user", "neo4j")
NEO4J_PASSWORD = _n4j.get("password", "example123")

# File Storage Configuration
_st = _cfg.get("storage", {})
DOCUMENTS_STORAGE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        _st.get("documents_dir", "../documents")
    )
)
