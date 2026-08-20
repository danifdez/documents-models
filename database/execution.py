import json
import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from config import EXECUTIONS_TABLE, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER

logger = logging.getLogger(__name__)


class Execution:
    def __init__(self):
        self.table = EXECUTIONS_TABLE
        self.connection_args = {
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "row_factory": dict_row,
        }
        self.conn = psycopg.connect(**self.connection_args, autocommit=True)

    def get_connection(self):
        return psycopg.connect(**self.connection_args, autocommit=True)

    @staticmethod
    def _canonical_json(value: Dict[str, Any]) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _append_event(
        self,
        cur,
        execution: Dict[str, Any],
        *,
        event_type: str,
        payload_schema: str,
        payload: Dict[str, Any],
        producer_instance_id: str,
        actor: Dict[str, Any],
        attempt_id: Optional[str] = None,
    ) -> str:
        root_execution_id = str(execution["root_execution_id"])
        cur.execute(
            f"SELECT last_sequence, last_event_id FROM {self.table} WHERE execution_id = %s FOR UPDATE",
            (root_execution_id,),
        )
        root = cur.fetchone()
        if not root:
            raise RuntimeError(f"Root execution {root_execution_id} not found")

        cur.execute(
            """
            SELECT COALESCE(MAX(producer_sequence), 0) + 1 AS next_sequence
            FROM execution_events
            WHERE root_execution_id = %s
              AND producer_component = 'documents-models'
              AND producer_instance_id = %s
            """,
            (root_execution_id, producer_instance_id),
        )
        producer_sequence = int(cur.fetchone()["next_sequence"])
        sequence = int(root["last_sequence"] or 0) + 1
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat().replace("+00:00", "Z")
        envelope = {
            "schemaVersion": "execution-event/1",
            "eventId": event_id,
            "rootExecutionId": root_execution_id,
            "executionId": str(execution["execution_id"]),
            "sequence": sequence,
            "producerSequence": producer_sequence,
            "eventType": event_type,
            "producer": {
                "component": "documents-models",
                "instanceId": producer_instance_id,
                "version": os.environ.get("MODELS_REVISION", "development"),
            },
            "actor": actor,
            "occurredAt": timestamp,
            "ingestedAt": timestamp,
            "payloadSchema": payload_schema,
            "payload": payload,
            "artifactRefs": [],
            "security": {
                "dataClassification": "workspace",
                "purpose": "evaluation",
                "allowedDestinations": ["documents", "ai-train"],
                "redactionApplied": False,
            },
        }
        if execution.get("parent_execution_id"):
            envelope["parentExecutionId"] = str(execution["parent_execution_id"])
        if execution.get("turn_id"):
            envelope["turnId"] = str(execution["turn_id"])
        if attempt_id:
            envelope["attemptId"] = str(attempt_id)
        if root.get("last_event_id"):
            envelope["causedByEventId"] = str(root["last_event_id"])
        content_hash = "sha256:" + hashlib.sha256(
            self._canonical_json(envelope).encode("utf-8")
        ).hexdigest()
        envelope["contentHash"] = content_hash

        cur.execute(
            """
            INSERT INTO execution_events (
              event_id, root_execution_id, sequence, producer_component,
              producer_instance_id, producer_sequence, event_type, execution_id,
              operation_id, attempt_id, caused_by_event_id, occurred_at,
              ingested_at, content_hash, envelope
            ) VALUES (%s, %s, %s, 'documents-models', %s, %s, %s, %s,
                      NULL, %s, %s, %s, %s, %s, %s)
            """,
            (
                event_id,
                root_execution_id,
                sequence,
                producer_instance_id,
                producer_sequence,
                event_type,
                execution["execution_id"],
                attempt_id,
                root.get("last_event_id"),
                now,
                now,
                content_hash,
                json.dumps(envelope),
            ),
        )
        cur.execute(
            f"UPDATE {self.table} SET last_sequence = %s, last_event_id = %s, updated_at = now() WHERE execution_id = %s",
            (sequence, event_id, root_execution_id),
        )
        return event_id

    def claim_pending_execution(self, worker_id: str, capabilities: List[str]) -> Optional[Dict[str, Any]]:
        from worker.capabilities import get_supported_task_types

        supported_types = get_supported_task_types(capabilities)
        if not supported_types:
            return None
        conn = psycopg.connect(**self.connection_args, autocommit=False)
        try:
            with conn.cursor() as cur:
                priorities = ["high", "normal"]
                if self._is_background_eligible(cur):
                    priorities.append("background")
                type_slots = ",".join(["%s"] * len(supported_types))
                priority_slots = ",".join(["%s"] * len(priorities))
                cur.execute(
                    f"""
                    SELECT * FROM {self.table}
                    WHERE status = 'queued' AND available_at <= now()
                      AND task_type IN ({type_slots})
                      AND priority IN ({priority_slots})
                    ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,
                             available_at, created_at
                    LIMIT 1 FOR UPDATE SKIP LOCKED
                    """,
                    (*supported_types, *priorities),
                )
                execution = cur.fetchone()
                if not execution:
                    conn.rollback()
                    return None
                attempt_id = str(uuid.uuid4())
                cur.execute(
                    f"""
                    UPDATE {self.table}
                    SET status = 'running', phase = 'worker_execution', claimed_by = %s,
                        attempt_id = %s, started_at = COALESCE(started_at, now()), updated_at = now()
                    WHERE execution_id = %s
                    """,
                    (worker_id, attempt_id, execution["execution_id"]),
                )
                execution.update(
                    status="running",
                    phase="worker_execution",
                    claimed_by=worker_id,
                    attempt_id=attempt_id,
                )
                self._append_event(
                    cur,
                    execution,
                    event_type="execution.state_changed",
                    payload_schema="execution.state_changed/1",
                    payload={
                        "from": "queued",
                        "to": "running",
                        "phase": "worker_execution",
                    },
                    producer_instance_id=f"queue:{worker_id}:{attempt_id}",
                    actor={"type": "worker", "id": worker_id},
                    attempt_id=attempt_id,
                )
                conn.commit()
                self._decode(execution)
                return execution
        except Exception:
            conn.rollback()
            logger.exception("Error claiming execution")
            return None
        finally:
            conn.close()

    def _is_background_eligible(self, cur) -> bool:
        from lib.llm.config import get_worker_config

        worker = get_worker_config()
        hour = datetime.now().hour
        if int(worker.get("background_hours_start", 2)) <= hour < int(worker.get("background_hours_end", 6)):
            return True
        cur.execute(f"SELECT COUNT(*) AS count FROM {self.table} WHERE status = 'queued' AND priority IN ('high', 'normal')")
        return cur.fetchone()["count"] == 0

    def update_execution_status(
        self,
        execution_id: str,
        status: str,
        phase: Optional[str] = None,
        attempt_id: Optional[str] = None,
        failure_message: Optional[str] = None,
    ) -> bool:
        terminal = status in {"completed", "failed", "cancelled"}
        released = status in {"queued", "waiting"} or phase == "backend_finalization" or terminal
        conn = psycopg.connect(**self.connection_args, autocommit=False)
        try:
            with conn.cursor() as cur:
                attempt_filter = " AND attempt_id = %s" if attempt_id else ""
                select_params = [execution_id]
                if attempt_id:
                    select_params.append(attempt_id)
                cur.execute(
                    f"SELECT * FROM {self.table} WHERE execution_id = %s{attempt_filter} FOR UPDATE",
                    select_params,
                )
                execution = cur.fetchone()
                if not execution:
                    conn.rollback()
                    return False
                self._decode(execution)
                previous_status = execution["status"]
                error = execution.get("error")
                completion_kind = execution.get("completion_kind")
                completion_reason = execution.get("completion_reason")
                if terminal:
                    completion_kind = "full"
                    completion_reason = {
                        "completed": "worker_completed",
                        "failed": "worker_failed",
                        "cancelled": "cancelled",
                    }[status]
                    if status == "failed" and not error:
                        error = {
                            "code": "EXECUTION_FAILED",
                            "message": failure_message or "Execution failed in worker",
                        }
                cur.execute(
                    f"""
                    UPDATE {self.table}
                    SET status = %s, phase = %s,
                        claimed_by = CASE WHEN %s THEN NULL ELSE claimed_by END,
                        attempt_id = CASE WHEN %s THEN NULL ELSE attempt_id END,
                        started_at = CASE WHEN %s THEN NULL ELSE started_at END,
                        completed_at = CASE WHEN %s THEN now() ELSE completed_at END,
                        completion_kind = %s,
                        completion_reason = %s,
                        error = %s,
                        updated_at = now()
                    WHERE execution_id = %s
                    """,
                    (
                        status,
                        phase,
                        released,
                        released,
                        released,
                        terminal,
                        completion_kind,
                        completion_reason,
                        json.dumps(error) if error is not None else None,
                        execution_id,
                    ),
                )
                payload = {"from": previous_status, "to": status}
                if phase:
                    payload["phase"] = phase
                if terminal:
                    payload.update(
                        completionKind=completion_kind,
                        completionReason=completion_reason,
                        result=execution.get("result"),
                        error=error,
                    )
                self._append_event(
                    cur,
                    execution,
                    event_type="execution.state_changed",
                    payload_schema="execution.state_changed/1",
                    payload=payload,
                    producer_instance_id=f"queue-transition:{execution_id}",
                    actor={"type": "worker"},
                    attempt_id=attempt_id,
                )
                conn.commit()
                return True
        except Exception:
            conn.rollback()
            logger.exception("Error updating execution %s", execution_id)
            return False
        finally:
            conn.close()

    def update_execution_result(self, execution_id: str, result: Dict[str, Any], result_blob: Optional[bytes] = None, attempt_id: Optional[str] = None) -> bool:
        try:
            with self.conn.cursor() as cur:
                attempt_filter = " AND attempt_id = %s" if attempt_id else ""
                params = [json.dumps(result), result_blob, execution_id]
                if attempt_id:
                    params.append(attempt_id)
                cur.execute(
                    f"UPDATE {self.table} SET result = %s, result_blob = COALESCE(%s, result_blob), updated_at = now() WHERE execution_id = %s{attempt_filter}",
                    params,
                )
                return cur.rowcount > 0
        except Exception:
            logger.exception("Error updating execution result %s", execution_id)
            return False

    def update_agent_progress(self, execution_id: str, step: int, checkpoint: Dict[str, Any], attempt_id: Optional[str] = None) -> bool:
        try:
            with self.conn.cursor() as cur:
                attempt_filter = " AND attempt_id = %s" if attempt_id else ""
                params = [step, json.dumps(checkpoint), execution_id]
                if attempt_id:
                    params.append(attempt_id)
                cur.execute(
                    f"""
                    UPDATE {self.table} SET step = %s, checkpoint = %s, updated_at = now()
                    WHERE execution_id = %s{attempt_filter}
                    """,
                    params,
                )
                return cur.rowcount > 0
        except Exception:
            logger.exception("Error updating execution checkpoint %s", execution_id)
            return False

    def update_agent_state(self, execution_id: str, checkpoint: Dict[str, Any], attempt_id: Optional[str] = None) -> bool:
        try:
            with self.conn.cursor() as cur:
                attempt_filter = " AND attempt_id = %s" if attempt_id else ""
                params = [json.dumps(checkpoint), execution_id]
                if attempt_id:
                    params.append(attempt_id)
                cur.execute(
                    f"UPDATE {self.table} SET checkpoint = %s, updated_at = now() WHERE execution_id = %s{attempt_filter}",
                    params,
                )
                return cur.rowcount > 0
        except Exception:
            logger.exception("Error updating execution checkpoint %s", execution_id)
            return False

    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self.table} WHERE execution_id = %s", (execution_id,))
                row = cur.fetchone()
                self._decode(row)
                return row
        except Exception:
            logger.exception("Error fetching execution %s", execution_id)
            return None

    def enqueue_child_execution(self, parent_execution_id: str, task_type: str, payload: Dict[str, Any], priority: str = "normal", agent_max_steps: int = 1, agent_kind: Optional[str] = None) -> Optional[str]:
        conn = psycopg.connect(**self.connection_args, autocommit=False)
        try:
            execution_id = str(uuid.uuid4())
            child_payload = dict(payload or {})
            if agent_kind:
                child_payload["kind"] = agent_kind
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM {self.table} WHERE execution_id = %s FOR UPDATE",
                    (parent_execution_id,),
                )
                parent = cur.fetchone()
                if not parent:
                    conn.rollback()
                    return None
                self._decode(parent)
                cur.execute(
                    f"""
                    INSERT INTO {self.table} (
                      execution_id, root_execution_id, parent_execution_id, owner_principal,
                      workspace_id, schema_version, task_type, origin, priority, payload, status, max_steps
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'child', %s, %s, 'queued', %s)
                    """,
                    (execution_id, parent["root_execution_id"], parent_execution_id,
                     parent["owner_principal"], parent["workspace_id"], parent["schema_version"],
                     task_type, priority, json.dumps(child_payload), agent_max_steps),
                )
                child = {
                    "execution_id": execution_id,
                    "root_execution_id": parent["root_execution_id"],
                    "parent_execution_id": parent_execution_id,
                    "turn_id": None,
                }
                self._append_event(
                    cur,
                    child,
                    event_type="execution.created",
                    payload_schema="execution.created/1",
                    payload={
                        "executionKind": task_type,
                        "initialStatus": "queued",
                    },
                    producer_instance_id=f"orchestrator:{parent_execution_id}",
                    actor={"type": "worker"},
                )
                conn.commit()
            return execution_id
        except Exception:
            conn.rollback()
            logger.exception("Error enqueueing child execution")
            return None
        finally:
            conn.close()

    def wake_waiting_execution(self, execution_id: str) -> bool:
        return self.update_execution_status(execution_id, "queued")

    @staticmethod
    def _build_retry_payload(checkpoint: Dict[str, Any], index: int) -> Dict[str, Any]:
        chunks = checkpoint.get("chunks") or []
        payload = dict(checkpoint.get("chunk_payload_template") or {})
        payload[checkpoint.get("chunk_field", "content")] = chunks[index] if index < len(chunks) else ""
        payload["_chunk_idx"] = index
        offsets = checkpoint.get("chunk_offsets")
        if isinstance(offsets, list) and index < len(offsets):
            payload["_chunk_offset"] = offsets[index]
        return payload

    def resume_parent_with_child(self, parent_id: str, child_id: str, *, success_result: Optional[Dict[str, Any]] = None, error: Optional[str] = None, max_retries: int = 0) -> Dict[str, Any]:
        parent = self.get_execution(parent_id)
        if not parent:
            return {"action": "ignored", "reason": "parent_not_found"}
        checkpoint = parent.get("checkpoint") or {}
        waiting = checkpoint.get("waiting_for_children") or {}
        key = str(child_id)
        if key not in waiting:
            return {"action": "ignored", "reason": "not_waiting_on_child"}
        index = int(waiting.pop(key))
        pending = checkpoint.setdefault("pending", {})
        pending.pop(key, None)
        results = checkpoint.setdefault("results", {})
        action = "result_recorded"
        if error is None:
            results[str(index)] = success_result or {}
        else:
            retries = checkpoint.setdefault("retries", {})
            attempts = int(retries.get(str(index), 0))
            child = None
            if attempts < max_retries:
                child = self.enqueue_child_execution(parent_id, parent["task_type"], self._build_retry_payload(checkpoint, index))
            if child:
                retries[str(index)] = attempts + 1
                waiting[child] = index
                pending[child] = index
                action = "retry_enqueued"
            else:
                results[str(index)] = None
                checkpoint.update(failed_idx=index, failed_error=error)
                action = "failed_no_retries"
        checkpoint["waiting_for_children"] = waiting
        self.update_agent_state(parent_id, checkpoint)
        if not waiting:
            self.wake_waiting_execution(parent_id)
        return {"action": action, "all_done": not waiting}

    @staticmethod
    def _decode(row: Optional[Dict[str, Any]]) -> None:
        if not row:
            return
        for key in ("payload", "result", "checkpoint"):
            if isinstance(row.get(key), str):
                try:
                    row[key] = json.loads(row[key])
                except Exception:
                    pass


_execution_database: Optional[Execution] = None


def get_execution_database() -> Execution:
    global _execution_database
    if _execution_database is None:
        _execution_database = Execution()
    return _execution_database
