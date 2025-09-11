import os
import json
import psycopg
from psycopg.rows import dict_row
from typing import Optional, Dict, Any


class Job:
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "database")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.db = os.getenv("POSTGRES_DB", "documents")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "example")
        self.table = os.getenv("JOBS_TABLE", "jobs")

        self.conn = psycopg.connect(
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
            priorities = ["high", "normal", "low"]
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
            print(f"Error fetching pending job: {e}")
            return None

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
            print(f"Error updating job status: {e}")
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
            print(f"Error updating job result: {e}")
            return False


# Singleton instance
_job_database: Optional[Job] = None


def get_job_database() -> Job:
    """Get the singleton Job service instance"""
    global _job_database
    if _job_database is None:
        _job_database = Job()
    return _job_database