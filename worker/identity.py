import json
import logging
import os
import uuid
import threading
import time

logger = logging.getLogger(__name__)

WORKER_ID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.worker_id')


def _load_or_create_worker_id() -> str:
    env_id = os.getenv("WORKER_ID")
    if env_id:
        return env_id
    try:
        with open(WORKER_ID_FILE, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        new_id = str(uuid.uuid4())
        with open(WORKER_ID_FILE, 'w') as f:
            f.write(new_id)
        return new_id


WORKER_ID = _load_or_create_worker_id()
WORKER_NAME = os.getenv("WORKER_NAME", f"worker-{WORKER_ID[:8]}")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "15"))


def register_worker(capabilities: list, metadata: dict):
    """Register or update this worker in the workers table."""
    from database.job import get_job_database

    db = get_job_database()
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO workers (id, name, capabilities, status, last_heartbeat, started_at, metadata)
                VALUES (%s, %s, %s::jsonb, 'online', NOW(), NOW(), %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    capabilities = EXCLUDED.capabilities,
                    status = 'online',
                    last_heartbeat = NOW(),
                    started_at = NOW(),
                    metadata = EXCLUDED.metadata
            """, (WORKER_ID, WORKER_NAME, json.dumps(capabilities), json.dumps(metadata)))
    finally:
        conn.close()


def start_heartbeat_thread():
    """Start a daemon thread that updates last_heartbeat every HEARTBEAT_INTERVAL seconds."""
    def _heartbeat_loop():
        while True:
            time.sleep(HEARTBEAT_INTERVAL)
            try:
                from database.job import get_job_database
                db = get_job_database()
                with db.conn.cursor() as cur:
                    cur.execute(
                        "UPDATE workers SET last_heartbeat = NOW() WHERE id = %s",
                        (WORKER_ID,)
                    )
            except Exception as e:
                logger.error("Heartbeat error: %s", e)

    t = threading.Thread(target=_heartbeat_loop, daemon=True)
    t.start()
    return t


def deregister_worker():
    """Mark this worker as offline on graceful shutdown."""
    try:
        from database.job import get_job_database
        db = get_job_database()
        with db.conn.cursor() as cur:
            cur.execute(
                "UPDATE workers SET status = 'offline' WHERE id = %s",
                (WORKER_ID,)
            )
    except Exception:
        pass
