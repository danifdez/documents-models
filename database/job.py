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
        from services.model_config import get_worker_config
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

    def update_job_result(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Update job result (stores JSON)"""
        try:
            result_json = json.dumps(result)
            with self.conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.table} SET result = %s WHERE id = %s",
                    (result_json, job_id),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.error("Error updating job result: %s", e)
            return False


# Singleton instance
_job_database: Optional[Job] = None


def get_job_database() -> Job:
    """Get the singleton Job service instance"""
    global _job_database
    if _job_database is None:
        _job_database = Job()
    return _job_database
