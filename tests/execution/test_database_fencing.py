import unittest
from unittest.mock import patch

from database.execution import Execution


class _Cursor:
    def __init__(self):
        self.rowcount = 1
        self.query = None
        self.params = None
        self.queries = []
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.query = query
        self.params = params
        self.queries.append((query, params))
        if "SELECT * FROM executions" in query:
            self._result = {
                "execution_id": "execution-id",
                "root_execution_id": "execution-id",
                "parent_execution_id": None,
                "turn_id": None,
                "status": "running",
                "result": None,
                "error": None,
                "completion_kind": None,
                "completion_reason": None,
            }
        elif "SELECT last_sequence" in query:
            self._result = {"last_sequence": 1, "last_event_id": "event-id"}
        elif "MAX(producer_sequence)" in query:
            self._result = {"next_sequence": 1}

    def fetchone(self):
        return self._result


class _Connection:
    def __init__(self):
        self.last_cursor = None

    def cursor(self):
        self.last_cursor = _Cursor()
        return self.last_cursor

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class ExecutionFencingTests(unittest.TestCase):
    def setUp(self):
        self.connection = _Connection()
        self.database = Execution.__new__(Execution)
        self.database.table = "executions"
        self.database.conn = self.connection
        self.database.connection_args = {}

    def test_worker_status_update_is_fenced_by_attempt(self):
        with patch("database.execution.psycopg.connect", return_value=self.connection):
            updated = self.database.update_execution_status(
                "execution-id",
                "waiting",
                attempt_id="attempt-id",
            )

        self.assertTrue(updated)
        cursor = self.connection.last_cursor
        select = next(item for item in cursor.queries if "SELECT * FROM executions" in item[0])
        self.assertIn("AND attempt_id = %s", select[0])
        self.assertEqual(select[1], ["execution-id", "attempt-id"])

    def test_worker_result_update_is_fenced_by_attempt(self):
        updated = self.database.update_execution_result(
            "execution-id",
            {"value": 1},
            attempt_id="attempt-id",
        )

        self.assertTrue(updated)
        cursor = self.connection.last_cursor
        self.assertIn("AND attempt_id = %s", cursor.query)
        self.assertEqual(cursor.params[-2:], ["execution-id", "attempt-id"])

    def test_checkpoint_update_preserves_claim_until_state_transition(self):
        self.database.update_agent_progress(
            "execution-id",
            2,
            {"cursor": 3},
            attempt_id="attempt-id",
        )

        cursor = self.connection.last_cursor
        set_clause = cursor.query.split("WHERE", 1)[0]
        self.assertNotIn("claimed_by", set_clause)
        self.assertNotIn("attempt_id", set_clause)
        self.assertIn("AND attempt_id = %s", cursor.query)

    def test_coordinator_update_does_not_require_an_attempt(self):
        self.database.update_agent_state("execution-id", {"cursor": 3})

        cursor = self.connection.last_cursor
        self.assertNotIn("AND attempt_id = %s", cursor.query)
        self.assertEqual(cursor.params[-1], "execution-id")


if __name__ == "__main__":
    unittest.main()
