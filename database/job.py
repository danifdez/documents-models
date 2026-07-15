import json
import logging
import os
import psycopg
from psycopg.rows import dict_row
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)
from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    JOBS_TABLE
)


class Job:
    def __init__(self):
        self.host = POSTGRES_HOST
        self.port = POSTGRES_PORT
        self.db = POSTGRES_DB
        self.user = POSTGRES_USER
        self.password = POSTGRES_PASSWORD
        self.table = JOBS_TABLE

        self.conn = psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.db,
            user=self.user,
            password=self.password,
            autocommit=True,
            row_factory=dict_row,
        )

    def get_connection(self):
        """Return a new database connection for direct queries."""
        return psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.db,
            user=self.user,
            password=self.password,
            autocommit=True,
            row_factory=dict_row,
        )

    def get_pending_job(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the oldest pending job by priority order: high > normal > low.

        Returns a dict or None.
        """
        try:
            priorities = ["high", "normal", "background"]
            with self.conn.cursor() as cur:
                for priority in priorities:
                    cur.execute(
                        f"""
                        SELECT * FROM {self.table}
                        WHERE status = 'pending' AND priority = %s
                        ORDER BY created_at ASC
                        LIMIT 1
                        """,
                        (priority,)
                    )
                    job = cur.fetchone()
                    if job:
                        # convert payload/result from json where needed
                        if isinstance(job.get("payload"), str):
                            try:
                                job["payload"] = json.loads(job["payload"])
                            except Exception:
                                pass
                        return job
            return None
        except Exception as e:
            logger.error("Error fetching pending job: %s", e)
            return None

    def claim_pending_job(self, worker_id: str, capabilities: List[str]) -> Optional[Dict[str, Any]]:
        """
        Atomically claim the highest-priority pending job this worker can handle.
        Uses SELECT FOR UPDATE SKIP LOCKED to prevent double-claiming.
        """
        from worker.capabilities import get_supported_task_types

        supported_types = get_supported_task_types(capabilities)
        if not supported_types:
            return None

        conn = psycopg.connect(
            host=self.host, port=self.port, dbname=self.db,
            user=self.user, password=self.password,
            row_factory=dict_row, autocommit=False
        )

        try:
            with conn.cursor() as cur:
                background_eligible = self._is_background_eligible(cur)

                eligible_priorities = ['high', 'normal']
                if background_eligible:
                    eligible_priorities.append('background')

                placeholders_types = ','.join(['%s'] * len(supported_types))
                placeholders_prio = ','.join(['%s'] * len(eligible_priorities))

                cur.execute(
                    f"""
                    SELECT * FROM {self.table}
                    WHERE status = 'pending'
                      AND type IN ({placeholders_types})
                      AND priority IN ({placeholders_prio})
                    ORDER BY
                        CASE priority
                            WHEN 'high' THEN 0
                            WHEN 'normal' THEN 1
                            WHEN 'background' THEN 2
                        END ASC,
                        created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                    """,
                    (*supported_types, *eligible_priorities)
                )

                job = cur.fetchone()
                if job:
                    cur.execute(
                        f"""
                        UPDATE {self.table}
                        SET status = 'processing',
                            claimed_by = %s,
                            started_at = NOW()
                        WHERE id = %s
                        """,
                        (worker_id, job['id'])
                    )
                    conn.commit()

                    if isinstance(job.get("payload"), str):
                        try:
                            job["payload"] = json.loads(job["payload"])
                        except Exception:
                            pass
                    return job
                else:
                    conn.rollback()
                    return None
        except Exception as e:
            conn.rollback()
            logger.error("Error claiming job: %s", e)
            return None
        finally:
            conn.close()

    def _is_background_eligible(self, cur) -> bool:
        """Check if background jobs should run now."""
        from lib.llm.config import get_worker_config
        worker = get_worker_config()
        bg_start = int(worker.get("background_hours_start", 2))
        bg_end = int(worker.get("background_hours_end", 6))
        current_hour = datetime.now().hour

        if bg_start <= current_hour < bg_end:
            return True

        cur.execute(
            f"""
            SELECT COUNT(*) as cnt FROM {self.table}
            WHERE status = 'pending' AND priority IN ('high', 'normal')
            """
        )
        row = cur.fetchone()
        return row['cnt'] == 0

    def requeue_stale_jobs(self, timeout_seconds: int = 60, max_retries: int = 3) -> int:
        """Reset jobs stuck in 'processing' with a dead worker back to pending."""
        try:
            with self.conn.cursor() as cur:
                # Requeue jobs whose worker heartbeat is stale
                cur.execute(
                    f"""
                    UPDATE {self.table} j
                    SET status = CASE
                            WHEN j.retry_count >= %s THEN 'failed'
                            ELSE 'pending'
                        END,
                        claimed_by = CASE
                            WHEN j.retry_count >= %s THEN j.claimed_by
                            ELSE NULL
                        END,
                        started_at = CASE
                            WHEN j.retry_count >= %s THEN j.started_at
                            ELSE NULL
                        END,
                        retry_count = j.retry_count + 1
                    FROM workers w
                    WHERE j.claimed_by = w.id
                      AND j.status = 'processing'
                      AND w.last_heartbeat < NOW() - INTERVAL '{timeout_seconds} seconds'
                    RETURNING j.id, j.status
                    """,
                    (max_retries, max_retries, max_retries)
                )
                rows = cur.fetchall()
                if rows:
                    for row in rows:
                        logger.info("Requeued stale job %s → %s", row['id'], row['status'])
                return len(rows)
        except Exception as e:
            logger.error("Error requeuing stale jobs: %s", e)
            return 0

    def update_job_status(self, job_id: str, status: str) -> bool:
        """Update job status"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.table} SET status = %s WHERE id = %s",
                    (status, job_id),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.error("Error updating job status: %s", e)
            return False

    def update_job_result(
        self,
        job_id: str,
        result: Dict[str, Any],
        result_blob: Optional[bytes] = None,
    ) -> bool:
        """Update job result (JSON) and optionally the binary result_blob column."""
        try:
            result_json = json.dumps(result)
            with self.conn.cursor() as cur:
                if result_blob is not None:
                    cur.execute(
                        f"UPDATE {self.table} SET result = %s, result_blob = %s WHERE id = %s",
                        (result_json, result_blob, job_id),
                    )
                else:
                    cur.execute(
                        f"UPDATE {self.table} SET result = %s WHERE id = %s",
                        (result_json, job_id),
                    )
                return cur.rowcount > 0
        except Exception as e:
            logger.error("Error updating job result: %s", e)
            return False

    def update_agent_progress(self, job_id, iteration: int, state: Dict[str, Any]) -> bool:
        """Persist a step of an agent run.

        Writes the new iteration counter and state, releases the worker by
        clearing claimed_by/started_at, so the next polling cycle can pick
        the job up from any worker.
        """
        try:
            state_json = json.dumps(state)
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.table}
                    SET agent_iteration = %s,
                        agent_state = %s,
                        claimed_by = NULL,
                        started_at = NULL
                    WHERE id = %s
                    """,
                    (iteration, state_json, job_id),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.error("Error updating agent progress: %s", e)
            return False

    def update_agent_state(self, job_id, state: Dict[str, Any]) -> bool:
        try:
            state_json = json.dumps(state)
            with self.conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.table} SET agent_state = %s WHERE id = %s",
                    (state_json, job_id),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.error("Error updating agent state: %s", e)
            return False

    def get_job(self, job_id) -> Optional[Dict[str, Any]]:
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self.table} WHERE id = %s", (job_id,))
                row = cur.fetchone()
                if row and isinstance(row.get("payload"), str):
                    try:
                        row["payload"] = json.loads(row["payload"])
                    except Exception:
                        pass
                return row
        except Exception as e:
            logger.error("Error fetching job %s: %s", job_id, e)
            return None

    def enqueue_child_job(
        self,
        parent_job_id,
        job_type: str,
        payload: Dict[str, Any],
        priority: str = "normal",
        agent_max_steps: int = 1,
        agent_kind: Optional[str] = None,
    ) -> Optional[int]:
        """Insert a new pending job linked to a parent. Returns the new id."""
        try:
            payload_json = json.dumps(payload or {})
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.table}
                        (type, priority, payload, status, parent_job_id, agent_max_steps, agent_kind)
                    VALUES (%s, %s, %s, 'pending', %s, %s, %s)
                    RETURNING id
                    """,
                    (job_type, priority, payload_json, parent_job_id, agent_max_steps, agent_kind),
                )
                row = cur.fetchone()
                return row["id"] if row else None
        except Exception as e:
            logger.error("Error enqueueing child job: %s", e)
            return None

    def wake_waiting_job(self, job_id) -> bool:
        """Move a job from 'waiting' back to 'pending' (after a child completes)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.table} SET status = 'pending' WHERE id = %s AND status = 'waiting'",
                    (job_id,),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.error("Error waking waiting job %s: %s", job_id, e)
            return False

    @staticmethod
    def _build_retry_payload(state: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Reconstruct the payload for a child being retried after failure.

        Uses `state["chunk_payload_template"]` (static fields copied as-is)
        plus `state["chunk_field"]` (where to put the chunk text). Adds
        `_chunk_idx` and, when present, `_chunk_offset` from
        `state["chunk_offsets"][idx]`. Falls back to the legacy summarize/
        key-point shape when no template is present.
        """
        chunks = state.get("chunks") or []
        chunk = chunks[idx] if idx < len(chunks) else ""
        template = state.get("chunk_payload_template")
        if isinstance(template, dict):
            payload = dict(template)
            chunk_field = state.get("chunk_field", "content")
            payload[chunk_field] = chunk
            payload["_chunk_idx"] = idx
            offsets = state.get("chunk_offsets")
            if isinstance(offsets, list) and idx < len(offsets):
                payload["_chunk_offset"] = offsets[idx]
            return payload
        # Legacy fallback: matches the old hardcoded summarize/key-point shape.
        payload = {
            "content": chunk,
            "targetLanguage": state.get("targetLanguage"),
            "_chunk_idx": idx,
        }
        if state.get("sourceLanguage") is not None:
            payload["sourceLanguage"] = state["sourceLanguage"]
        return payload

    def resume_parent_with_child(
        self,
        parent_id,
        child_id,
        *,
        success_result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        max_retries: int = 0,
    ) -> Dict[str, Any]:
        """Atomically apply a child completion to a parent waiting on N children.

        Locks the parent row, reads `agent_state.waiting_for_children`, removes the
        child id, records its result (or applies a retry by enqueueing a new child),
        writes back the state, and wakes the parent if no children remain.

        Returns a dict describing what happened:
          - {"action": "ignored", "reason": ...}
          - {"action": "result_recorded", "all_done": bool}
          - {"action": "retry_enqueued", "new_child_id": int}
          - {"action": "failed_no_retries", "all_done": bool}
        """
        conn = psycopg.connect(
            host=self.host, port=self.port, dbname=self.db,
            user=self.user, password=self.password,
            row_factory=dict_row, autocommit=False,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT agent_state, status, type FROM {self.table} WHERE id = %s FOR UPDATE",
                    (parent_id,),
                )
                row = cur.fetchone()
                if not row:
                    conn.rollback()
                    return {"action": "ignored", "reason": "parent_not_found"}

                state = row.get("agent_state") or {}
                if isinstance(state, str):
                    try:
                        state = json.loads(state)
                    except Exception:
                        state = {}

                waiting = state.get("waiting_for_children") or {}
                key = str(child_id)
                if key not in waiting:
                    conn.rollback()
                    return {"action": "ignored", "reason": "not_waiting_on_child"}

                idx = int(waiting[key])
                results = state.setdefault("results", {})
                retries = state.setdefault("retries", {})
                pending = state.setdefault("pending", {})
                idx_key = str(idx)

                if error is not None:
                    attempts = int(retries.get(idx_key, 0))
                    if attempts < max_retries:
                        chunks = state.get("chunks") or []
                        if idx >= len(chunks):
                            conn.rollback()
                            return {"action": "ignored", "reason": "chunk_missing"}
                        retry_payload = self._build_retry_payload(state, idx)
                        cur.execute(
                            f"""
                            INSERT INTO {self.table}
                                (type, priority, payload, status, parent_job_id, agent_max_steps, agent_kind)
                            VALUES (%s, %s, %s, 'pending', %s, %s, %s)
                            RETURNING id
                            """,
                            (row["type"], "normal", json.dumps(retry_payload),
                             parent_id, 1, None),
                        )
                        new_row = cur.fetchone()
                        new_id = new_row["id"] if new_row else None
                        if new_id is None:
                            conn.rollback()
                            return {"action": "ignored", "reason": "retry_enqueue_failed"}
                        retries[idx_key] = attempts + 1
                        waiting.pop(key, None)
                        waiting[str(new_id)] = idx
                        pending.pop(key, None)
                        pending[str(new_id)] = idx
                        state["waiting_for_children"] = waiting
                        state["pending"] = pending
                        state["retries"] = retries
                        cur.execute(
                            f"UPDATE {self.table} SET agent_state = %s WHERE id = %s",
                            (json.dumps(state), parent_id),
                        )
                        conn.commit()
                        return {"action": "retry_enqueued", "new_child_id": new_id}
                    else:
                        results[idx_key] = None
                        state["failed_idx"] = idx
                        state["failed_error"] = error
                        waiting.pop(key, None)
                        pending.pop(key, None)
                        action = "failed_no_retries"
                else:
                    results[idx_key] = success_result if isinstance(success_result, dict) else {}
                    waiting.pop(key, None)
                    pending.pop(key, None)
                    action = "result_recorded"

                state["waiting_for_children"] = waiting
                state["pending"] = pending
                state["results"] = results
                cur.execute(
                    f"UPDATE {self.table} SET agent_state = %s WHERE id = %s",
                    (json.dumps(state), parent_id),
                )

                all_done = not waiting
                if all_done:
                    cur.execute(
                        f"UPDATE {self.table} SET status = 'pending' WHERE id = %s AND status = 'waiting'",
                        (parent_id,),
                    )

                conn.commit()
                return {"action": action, "all_done": all_done}
        except Exception as e:
            conn.rollback()
            logger.error("Error resuming parent %s with child %s: %s", parent_id, child_id, e)
            return {"action": "ignored", "reason": f"exception: {e}"}
        finally:
            conn.close()


# Singleton instance
_job_database: Optional[Job] = None


def get_job_database() -> Job:
    """Get the singleton Job service instance"""
    global _job_database
    if _job_database is None:
        _job_database = Job()
    return _job_database
