"""pgvector-backed vector store.

Each logical collection maps to its own PostgreSQL table (physical isolation):

- ``rag_chunks``           — workspace RAG (resources / docs / knowledge). Sources
                             are heterogeneous, so cleanup is app-driven by
                             ``source_id`` (no FK).
- ``indexed_file_chunks``  — assistant working-folder files. FK to ``indexed_files``
                             with ``ON DELETE CASCADE``; ``owner_tag`` promoted to a
                             column for strict, indexed owner filtering.
- ``memory_vectors``       — assistant personal memory, 1-to-1 with
                             ``assistant_memory_entries`` (FK + CASCADE); ``memory_id``
                             is the primary key so re-ingest upserts in place.

Each table promotes a few payload keys to real, indexed columns (FK + fast
filtering) and also stores the full payload as JSONB, so queries return it intact.
Vectors are L2-normalized E5 (384-dim); similarity is cosine, so the score is
``1 - (embedding <=> query)`` (1.0 = identical).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector

from config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD,
    RAG_TABLE, FOLDER_TABLE, MEMORY_TABLE,
)

logger = logging.getLogger(__name__)

# Columns stored as integers (everything else promoted is text).
INT_COLUMNS = {"indexed_file_id", "memory_id"}


@dataclass
class PointStruct:
    """A vector point to upsert: id, embedding and an arbitrary payload."""
    id: Any
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredPoint:
    """A scored search hit: exposes ``.score`` and ``.payload``."""
    score: float
    payload: Dict[str, Any]


class Rag:
    """Thin wrapper around a single pgvector-backed table.

    The public surface (``upsert_points`` / ``query_points`` / ``delete_by_column``
    / ``delete_points`` / ``recreate_collection``) is table-agnostic; per-table
    differences (primary key, promoted columns) are passed to the constructor.
    """

    def __init__(self, table: str, pk_column: str, promoted: Dict[str, str]):
        # ``promoted`` maps db_column -> payload key. The PK is taken from
        # ``point.id`` unless ``pk_column`` is itself promoted (memory takes its
        # PK ``memory_id`` from the payload).
        self.table = table
        self.pk_column = pk_column
        self.promoted = promoted
        self.columns = set(promoted) | {pk_column}
        self.conn = psycopg.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT, dbname=POSTGRES_DB,
            user=POSTGRES_USER, password=POSTGRES_PASSWORD,
            autocommit=True, row_factory=dict_row,
        )
        register_vector(self.conn)
        # pgvector >= 0.8: keep recall when combining ANN search with WHERE
        # filters (project_id / owner_tag / assistant_id). No-op on older builds.
        try:
            self.conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
        except psycopg.Error:
            logger.info("hnsw.iterative_scan unavailable (pgvector < 0.8); continuing")

    def _cast(self, column: str, raw: Any) -> Any:
        if raw is None:
            return None
        return int(raw) if column in INT_COLUMNS else str(raw)

    def _pk_value(self, point: PointStruct) -> Any:
        if self.pk_column in self.promoted:
            return self._cast(self.pk_column, point.payload.get(self.promoted[self.pk_column]))
        return point.id

    def upsert_points(self, points: List[PointStruct]) -> bool:
        if not points:
            return True
        extra = [c for c in self.promoted if c != self.pk_column]
        cols = [self.pk_column, "embedding"] + extra + ["payload"]
        insert_cols = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join(["%s"] * len(cols))
        updates = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c != self.pk_column)
        sql = (
            f'INSERT INTO "{self.table}" ({insert_cols}) VALUES ({placeholders}) '
            f'ON CONFLICT ("{self.pk_column}") DO UPDATE SET {updates}'
        )
        rows = []
        for p in points:
            row = [self._pk_value(p), p.vector]
            for c in extra:
                row.append(self._cast(c, p.payload.get(self.promoted[c])))
            row.append(Jsonb(p.payload))
            rows.append(row)
        try:
            with self.conn.cursor() as cur:
                cur.executemany(sql, rows)
            return True
        except psycopg.Error as e:
            logger.error("Error upserting into %s: %s", self.table, e)
            return False

    def query_points(self, query_vector: List[float], limit: int = 3, with_payload: bool = True,
                     project_id: Optional[str] = None, assistant_id: Optional[str] = None,
                     owner_tag: Optional[str] = None,
                     score_threshold: Optional[float] = None) -> List[ScoredPoint]:
        params: Dict[str, Any] = {"qv": query_vector, "limit": int(limit)}
        where: List[str] = []
        for col, val in (("project_id", project_id), ("assistant_id", assistant_id),
                         ("owner_tag", owner_tag)):
            if val is not None and col in self.columns:
                params[col] = self._cast(col, val)
                where.append(f'"{col}" = %({col})s')
        if score_threshold is not None:
            params["threshold"] = float(score_threshold)
            where.append("1 - (embedding <=> %(qv)s) >= %(threshold)s")
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = (
            f'SELECT payload, 1 - (embedding <=> %(qv)s) AS score '
            f'FROM "{self.table}"{where_sql} '
            f'ORDER BY embedding <=> %(qv)s LIMIT %(limit)s'
        )
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                return [ScoredPoint(score=float(r["score"]), payload=r["payload"] or {})
                        for r in cur.fetchall()]
        except psycopg.Error as e:
            logger.error("Error querying %s: %s", self.table, e)
            return []

    def delete_by_column(self, column: str, value: Any) -> bool:
        """Delete every row where ``column`` matches ``value``. The column must be
        one of the table's promoted/PK columns."""
        if column not in self.columns:
            logger.warning("delete_by_column: %s has no column %s", self.table, column)
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute(f'DELETE FROM "{self.table}" WHERE "{column}" = %s',
                            (self._cast(column, value),))
            return True
        except psycopg.Error as e:
            logger.error("Error deleting from %s by %s: %s", self.table, column, e)
            return False

    def delete_points(self, ids: List[Any]) -> bool:
        """Delete rows by primary key."""
        if not ids:
            return True
        try:
            cast_ids = [self._cast(self.pk_column, i) for i in ids]
            with self.conn.cursor() as cur:
                cur.execute(f'DELETE FROM "{self.table}" WHERE "{self.pk_column}" = ANY(%s)',
                            (cast_ids,))
            return True
        except psycopg.Error as e:
            logger.error("Error deleting points from %s: %s", self.table, e)
            return False

    def recreate_collection(self) -> bool:
        """Empty the table (used by reset/admin paths)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f'TRUNCATE TABLE "{self.table}"')
            return True
        except psycopg.Error as e:
            logger.error("Error truncating %s: %s", self.table, e)
            return False


_rag_database: Optional[Rag] = None
_folder_rag_database: Optional[Rag] = None
_memory_rag_database: Optional[Rag] = None


def get_rag() -> Rag:
    """Workspace RAG table (resources / docs / knowledge / notes)."""
    global _rag_database
    if _rag_database is None:
        _rag_database = Rag(
            table=RAG_TABLE,
            pk_column="id",
            promoted={"source_type": "source_type", "source_id": "source_id",
                      "project_id": "project_id"},
        )
    return _rag_database


def get_folder_rag() -> Rag:
    """Assistant working-folder table. FK to ``indexed_files`` (CASCADE); owner
    isolation is enforced by the ``owner_tag`` column filter."""
    global _folder_rag_database
    if _folder_rag_database is None:
        _folder_rag_database = Rag(
            table=FOLDER_TABLE,
            pk_column="id",
            promoted={"indexed_file_id": "indexed_file_id", "owner_tag": "owner_tag"},
        )
    return _folder_rag_database


def get_memory_rag() -> Rag:
    """Assistant personal-memory table. 1-to-1 with ``assistant_memory_entries``
    (FK + CASCADE); ``memory_id`` is the PK so updates upsert in place."""
    global _memory_rag_database
    if _memory_rag_database is None:
        _memory_rag_database = Rag(
            table=MEMORY_TABLE,
            pk_column="memory_id",
            promoted={"memory_id": "memory_id", "assistant_id": "assistant_id"},
        )
    return _memory_rag_database
