"""Direct relational access to ``assistant_memory_entries``.

Separate from ``database/rag.py`` (vector search over ``memory_vectors``):
this module reads the canonical rows (name/type/body) the vectors point to.
"""

import psycopg
from psycopg.rows import dict_row

from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)


def _memory_db_connection():
    # Same effective parameters as Execution.get_connection(), but kept local:
    # reusing it would instantiate the Execution singleton, which opens an extra
    # persistent connection as a side effect.
    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        autocommit=True,
        row_factory=dict_row,
    )


def recent_entries(assistant_id: int, limit: int) -> list:
    """Most recent entries for an assistant, newest first."""
    with _memory_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, type, body FROM assistant_memory_entries "
            "WHERE assistant_id = %s ORDER BY created_at DESC, id DESC LIMIT %s",
            (assistant_id, limit),
        )
        return cur.fetchall()


def entries_by_ids(assistant_id: int, ids: list) -> dict:
    """Entries owned by an assistant, keyed by id. Empty ids -> empty dict."""
    if not ids:
        return {}
    with _memory_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, type, body FROM assistant_memory_entries "
            "WHERE assistant_id = %s AND id = ANY(%s)",
            (assistant_id, list(ids)),
        )
        return {r["id"]: r for r in cur.fetchall()}
