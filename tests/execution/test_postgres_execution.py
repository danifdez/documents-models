import os
import unittest
import uuid
from unittest.mock import patch

import psycopg
from psycopg.rows import dict_row

from config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from database.execution import Execution


@unittest.skipUnless(
    os.environ.get("RUN_EXECUTION_POSTGRES_E2E") == "1",
    "set RUN_EXECUTION_POSTGRES_E2E=1 to run PostgreSQL integration tests",
)
class ExecutionPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = f"models_execution_test_{uuid.uuid4().hex}"
        cls.base_args = {
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "row_factory": dict_row,
        }
        cls.admin = psycopg.connect(**cls.base_args, autocommit=True)
        cls.admin.execute(f'CREATE SCHEMA "{cls.schema}"')
        cls.connection_args = {
            **cls.base_args,
            "options": f"-c search_path={cls.schema}",
        }
        with psycopg.connect(**cls.connection_args, autocommit=True) as connection:
            connection.execute(
                """
                CREATE TABLE executions (
                  execution_id uuid PRIMARY KEY,
                  root_execution_id uuid NOT NULL,
                  parent_execution_id uuid,
                  turn_id uuid,
                  owner_principal varchar(200) NOT NULL,
                  workspace_id varchar(200) NOT NULL,
                  schema_version varchar(50) NOT NULL,
                  task_type varchar(100) NOT NULL,
                  origin varchar(30) NOT NULL DEFAULT 'root',
                  priority varchar(20) NOT NULL DEFAULT 'normal',
                  payload jsonb,
                  status varchar(20) NOT NULL DEFAULT 'queued',
                  phase varchar(80),
                  wait_reason varchar(100),
                  completion_kind varchar(20),
                  completion_reason varchar(100),
                  result jsonb,
                  error jsonb,
                  checkpoint jsonb,
                  step integer NOT NULL DEFAULT 0,
                  max_steps integer NOT NULL DEFAULT 1,
                  available_at timestamptz NOT NULL DEFAULT now(),
                  claimed_by uuid,
                  attempt_id uuid,
                  retry_count integer NOT NULL DEFAULT 0,
                  max_attempts integer NOT NULL DEFAULT 3,
                  started_at timestamptz,
                  completed_at timestamptz,
                  input_blob bytea,
                  result_blob bytea,
                  last_sequence bigint NOT NULL DEFAULT 0,
                  last_event_id uuid,
                  completeness_status varchar(30) NOT NULL DEFAULT 'reproducible',
                  missing_evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  updated_at timestamptz NOT NULL DEFAULT now()
                );
                CREATE TABLE execution_events (
                  event_id uuid PRIMARY KEY,
                  root_execution_id uuid NOT NULL,
                  sequence bigint NOT NULL,
                  producer_component varchar(80) NOT NULL,
                  producer_instance_id varchar(200) NOT NULL,
                  producer_sequence bigint NOT NULL,
                  event_type varchar(80) NOT NULL,
                  execution_id uuid NOT NULL,
                  operation_id uuid,
                  attempt_id uuid,
                  caused_by_event_id uuid,
                  occurred_at timestamptz NOT NULL,
                  ingested_at timestamptz NOT NULL,
                  content_hash varchar(71) NOT NULL,
                  envelope jsonb NOT NULL,
                  UNIQUE (root_execution_id, sequence),
                  UNIQUE (
                    root_execution_id,
                    producer_component,
                    producer_instance_id,
                    producer_sequence
                  )
                );
                """
            )

    @classmethod
    def tearDownClass(cls):
        cls.admin.execute(f'DROP SCHEMA IF EXISTS "{cls.schema}" CASCADE')
        cls.admin.close()

    def setUp(self):
        self.database = Execution.__new__(Execution)
        self.database.table = "executions"
        self.database.connection_args = self.connection_args
        self.database.conn = psycopg.connect(
            **self.connection_args,
            autocommit=True,
        )

    def tearDown(self):
        self.database.conn.close()

    def test_claim_and_fenced_finalization_share_the_event_log(self):
        execution_id = str(uuid.uuid4())
        worker_id = str(uuid.uuid4())
        with self.database.conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO executions (
                  execution_id, root_execution_id, owner_principal,
                  workspace_id, schema_version, task_type, payload
                ) VALUES (%s, %s, 'test', 'test', 'execution-event/1',
                          'translate', '{}'::jsonb)
                """,
                (execution_id, execution_id),
            )

        with patch(
            "worker.capabilities.get_supported_task_types",
            return_value=["translate"],
        ):
            claimed = self.database.claim_pending_execution(worker_id, ["cpu"])

        self.assertEqual(str(claimed["execution_id"]), execution_id)
        attempt_id = str(claimed["attempt_id"])
        self.assertFalse(
            self.database.update_execution_result(
                execution_id,
                {"translated": "stale"},
                attempt_id=str(uuid.uuid4()),
            )
        )
        self.assertTrue(
            self.database.update_execution_result(
                execution_id,
                {"translated": "ok"},
                attempt_id=attempt_id,
            )
        )
        self.assertTrue(
            self.database.update_execution_status(
                execution_id,
                "running",
                phase="backend_finalization",
                attempt_id=attempt_id,
            )
        )

        stored = self.database.get_execution(execution_id)
        self.assertEqual(stored["result"], {"translated": "ok"})
        self.assertEqual(stored["phase"], "backend_finalization")
        self.assertIsNone(stored["attempt_id"])
        with self.database.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT envelope FROM execution_events
                WHERE root_execution_id = %s ORDER BY sequence
                """,
                (execution_id,),
            )
            events = [row["envelope"] for row in cursor.fetchall()]
        self.assertEqual(
            [event["payload"]["to"] for event in events],
            ["running", "running"],
        )
        self.assertEqual(events[0]["payload"]["from"], "queued")
        self.assertEqual(events[1]["payload"]["phase"], "backend_finalization")


if __name__ == "__main__":
    unittest.main()
